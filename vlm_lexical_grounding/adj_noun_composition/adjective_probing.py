import argparse
import random
import os
import json
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import transformers

from vlm_lexical_grounding.utils.simple_classifier import AdjectiveProbingHead
from vlm_lexical_grounding.utils.general_tools import get_embedder_path, set_seed

device = "cuda" if torch.cuda.is_available() else "cpu"



def parseArguments(): 
    parser = argparse.ArgumentParser()

    # Necessary variables
    parser.add_argument("--embedder", type=str, required=True)
    parser.add_argument("--cls_type", type=str, default="linear")
    parser.add_argument("--num_rows", type=int, default=-1,
                        help="Number of samples for finetuning. -1 means all samples.")
    parser.add_argument("--seed", type=int, default=1123)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report_step", type=int, default=100)
    
    parser.add_argument("--method", type=str, default="kmeans")
    parser.add_argument("--cluster_type", type=str, required=True,
                        help="The granularity of cluster, either adjective or noun or bigram.")
    parser.add_argument("--bigram_occur_threshold", type=int, default=10)
    parser.add_argument("--unique_bigram_threshold", type=int, default=10)
    parser.add_argument("--sample_bigram_per_target", type=int, default=-1)
    parser.add_argument("--sample_threshold", type=int, default=20)
    
    # Necessary training parameters
    parser.add_argument("--input_size", type=int, default=768)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--hidden_dropout", type=float, default=0.0)

    parser.add_argument("--train_step", type=int, default=1000)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()

    # I/O parameters
    df_info_path, embeddings_path = get_data_path(args)
    parser.add_argument("--data_path", type=str, default="../../data/wikiHow/wikihowAll_clean_single.csv")
    parser.add_argument("--mit_states_path", type=str, default="../../data/mit_states/release_dataset/adj_ants.csv")
    parser.add_argument("--general_statistics_path", type=str, default="../../outputs/general_statistics/general_statistics_Rall/df_general.csv")
    parser.add_argument("--df_info_path", type=str, default=df_info_path)
    parser.add_argument("--embeddings_path", type=str, default=embeddings_path)

    # Output parameter
    output_name = get_output_name(args)
    parser.add_argument("--output_name", type=str, default=output_name)
    args = parser.parse_args()

    return args

def get_output_name(args):
    output_name = "{}_{}_K{}_S{}_B{}_LR{}".format(
        args.embedder, 
        args.cls_type, 
        args.kfold, 
        args.train_step, 
        args.batch_size, 
        args.learning_rate
    )
    return output_name

def get_data_path(args):
    name = "{}_B{}_U{}".format(
        args.embedder,
        args.bigram_occur_threshold,
        args.unique_bigram_threshold,
    )
    if args.cluster_type=="ADJ":
        emb_name = "adj"
    elif args.cluster_type=="NOUN":
        emb_name = "noun"
    df_info_path = "../../outputs/get_target_embs/{}/df_info.csv".format(name)
    embeddings_path = "../../outputs/get_target_embs/{}/{}_embeddings.npy".format(name, emb_name)
    return df_info_path, embeddings_path

def get_logger(args):
    os.makedirs("../../logs/adjective_probing", exist_ok=True)
    os.makedirs("../../logs/adjective_probing/{}".format(args.output_name), exist_ok=True)
    logging.basicConfig(level=logging.INFO, \
            format = '%(asctime)s %(levelname)s: %(message)s', \
            datefmt = '%m/%d %H:%M:%S %p', \
            filename = '../../logs/adjective_probing/{}/{}.log'.format(args.output_name, args.output_name), \
            filemode = 'w'
    )
    return logging.getLogger(__name__)

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

def output_predictions(true_list, best_pred_list, valid_df_info, run_id, args):
    output_path = "../../outputs/adjective_probing/{}".format(args.output_name)
    logger.info("Output predictions to {}".format(output_path))
    os.makedirs(output_path, exist_ok=True)
    
    true_list = true_list.reshape([-1])
    best_pred_list = best_pred_list.reshape([-1])
    valid_df_info["label"] = true_list
    valid_df_info["pred"] = best_pred_list
    valid_df_info.to_csv("{}/run{}_df_info.csv".format(output_path, run_id))

def get_target_df(df, adj_ant_dict, args):
    # Bigram occurrence filter
    target_df = df[df.bigram_occur>=args.bigram_occur_threshold]

    # POS filter
    pos_filter = "NOUN" if args.cluster_type=="NOUN" else "ADJ"
    target_df = target_df[target_df.POS==pos_filter]
    
    # string filter for both target and bigram
    condition1 = target_df.target.str.isalpha()
    condition2 = target_df.bigram.str.isalpha()
    target_df = target_df[condition1 & condition2]

    # MIT states adjectives filter
    if args.cluster_type == "ADJ":
        target_df = target_df[target_df.target.isin(adj_ant_dict.keys())]
    elif args.cluster_type == "NOUN":
        target_df = target_df[target_df.bigram.isin(adj_ant_dict.keys())]

    # Unique bigram filter
    pass_unique_threshold_list = (target_df.groupby(["target"]).size()>=args.unique_bigram_threshold).to_dict()
    target_df.loc[:, "pass_unique_threshold"] = False
    for row_id in target_df.index:
        target = target_df.target[row_id]
        if (target in pass_unique_threshold_list) and pass_unique_threshold_list[target]:
            target_df.loc[row_id, "pass_unique_threshold"] = True
    target_df = target_df[target_df.pass_unique_threshold]

    target_df.reset_index(inplace=True)
    return target_df

def get_mit_states_data(args):
    adj_ant_dict = {}
    with open(args.mit_states_path, "r") as f:
        lines = f.readlines()
        for line_id in range(1, len(lines)):
            # Skip the column row
            line = lines[line_id].strip("\n")
            line = line.split(",")
            adj = line[0]
            ant = [word for word in line[1:] if len(word)>0]
            adj_ant_dict[adj] = ant
    return adj_ant_dict

def eval(model, data, valid_id, args):
    model.eval()
    true_list = []
    pred_list = []
    loader = DataLoader(data, shuffle=False, pin_memory=True, batch_size=args.batch_size)
    
    with torch.no_grad():
        for iter_id, batch in enumerate(loader):
            embeddings, labels = tuple(t.to(device) for t in batch)
            logits, _ = model(embeddings, labels)
            
            logits = logits.detach().cpu().numpy()
            labels = labels.cpu().numpy()

            preds = np.argmax(logits, axis=1)
            pred_list.append(preds)
            true_list.append(labels)
            
    pred_list = np.concatenate(pred_list)
    true_list = np.concatenate(true_list)
    
    accuracy = (pred_list == true_list).sum() / true_list.size
    
    logger.info(f"| Validation {valid_id:6d} | eval_acc {accuracy:8.5f} |")
    return accuracy, pred_list, true_list

class AdjectiveDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return self.embeddings.shape[0]
    
    def __getitem__(self, idx):
        embeddings = torch.from_numpy(self.embeddings[idx])    
        label = torch.tensor(self.labels[idx]).long()
        return embeddings, label

def main(args):
    # Load dataset
    adj_ant_dict = get_mit_states_data(args)
    df = pd.read_csv(args.general_statistics_path)
    df = get_target_df(df, adj_ant_dict, args)
    df_info = pd.read_csv(args.df_info_path)
    precompute_embs = np.load(args.embeddings_path)

    # MIT states adjective filter
    df_info = df_info[df_info.adjective.isin(df.bigram.unique())]

    # Shuffle df_info and embeddings
    df_info = df_info.sample(frac=1, random_state=args.seed)

    # Convert adjective to adj_id
    adj_id_map = {adj:adj_id for adj_id, adj in enumerate(df_info.adjective.unique())}
    df_info.loc[:, "adjective_id"] = [adj_id_map[adj] for adj in df_info.adjective]

    output_size = len(df_info.adjective.unique())
    fold_size = len(df_info) // args.kfold

    accuracy_list = []
        
    for run_id in range(args.kfold):
        valid_df_info = df_info[run_id*fold_size:(run_id+1)*fold_size].copy()
        train_df_info = df_info[~df_info.index.isin(valid_df_info.index)].copy()
        valid_embeddings = precompute_embs[valid_df_info.index]
        train_embeddings = precompute_embs[train_df_info.index]
        valid_labels = valid_df_info.adjective_id.reset_index(drop=True)
        train_labels = train_df_info.adjective_id.reset_index(drop=True)
        assert len(train_df_info.adjective_id.unique()) == output_size
        
        train_dataset = AdjectiveDataset(train_embeddings, train_labels)
        valid_dataset = AdjectiveDataset(valid_embeddings, valid_labels)
        
        # Initialize probing classifier
        model = AdjectiveProbingHead(
            input_size=args.input_size, 
            output_size=output_size, 
            hidden_size=args.hidden_size, 
            hidden_dropout=args.hidden_dropout,
        )
        model.to(device)
        
        # Create optimizer
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode   = "max", 
            factor = args.factor,
            patience  = args.lr_patience,
            threshold = args.threshold,
            threshold_mode = "abs",
            min_lr  = args.min_lr,
            verbose = False
        )
        best_accuracy = float("-inf")

        loader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=args.batch_size, drop_last=True)
        loader_iter = iter(loader)
        best_valid_id = 0
        valid_id  = 0
        patience = 0
        lr_patience = 0
        best_pred_list = None

        while patience < args.patience:
            logger.info("="*70)
            
            # Training for args.eval_steps
            model.train()
            for iter_id in range(args.train_step):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize loader_iter
                    loader_iter = iter(loader)
                    batch = next(loader_iter)
                   
                embeddings, labels = tuple(t.to(device) for t in batch)
                logits, loss = model(embeddings, labels)
                loss.backward()
                optimizer.step()
                model.zero_grad()
                
                if args.verbose and iter_id % args.report_step == 0:
                    logger.info(f"| Validation {valid_id:6d} | iter {iter_id:8d} | train_loss {loss.item():8.5f} |")
                    
            # Early stopping and Evaluation
            accuracy, pred_list, true_list = eval(model, valid_dataset, valid_id, args)
            if accuracy > best_accuracy:
                logger.info("*"*70)
                best_accuracy = accuracy
                logger.info(f"Best model found with accuracy {float(accuracy):8.5f}")
                patience = 0
                lr_patience = 0
                best_valid_id = valid_id
                best_pred_list = pred_list
                logger.info("*"*70)
            else:
                patience += 1
                lr_patience = 0 if lr_patience == args.lr_patience else lr_patience + 1

            scheduler.step(accuracy)    
            logger.info(f"Current LR: {scheduler._last_lr[0]:e}")
            logger.info(f"Best accuracy {best_accuracy:8.5f} at Step {best_valid_id:d}")
            logger.info(f"Validation passes without improvement for this LR: {lr_patience:6d}")
            logger.info(f"Total validation passes without improvement: {patience:6d}")
                
            valid_id += 1
            
            logger.info("="*70)

        accuracy_list.append(best_accuracy)   
        total_steps  = valid_id * args.train_step
        total_epochs = total_steps / (len(train_dataset) / args.batch_size)
        logger.info(f"Stopped training after {valid_id:d} validation checks")
        logger.info(f"Trained for {total_steps:d} steps (batches) or {total_epochs:f} epochs")
        logger.info(f"Best model found with accuracy {float(best_accuracy):8.5f} at Validation {best_valid_id}")

        output_predictions(true_list, best_pred_list, valid_df_info, run_id, args)

    logger.info("#"*70)
    logger.info("#"*70)

    accuracy_list = np.array(accuracy_list)
    mean_accuracy = np.mean(accuracy_list)
    std_accuracy  = np.std(accuracy_list)
    accuracy_list = [str(round(acc, 4)) for acc in accuracy_list]
    str_accuracy_list = ", ".join(accuracy_list)
    logger.info(f"{args.kfold:4d} runs accuracy list: [{accuracy_list}]")
    logger.info(f"Mean accuracy: {mean_accuracy:8.4f}")
    logger.info(f"Standard deviation accuracy: {std_accuracy:8.4f}")
    logger.info("Done!")

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)
    set_seed(args.seed)

    main(args)
