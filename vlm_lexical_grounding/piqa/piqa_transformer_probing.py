import argparse
import os
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..utils.simple_classifier import MultipleChoiceProbingHead
from ..utils.general_tools import set_seed, get_embedder_path
from .piqa_tools import get_data_path, load_embedder
from .piqa_data import PIQADataset, get_tokenized_samples, load_data

device = "cuda" if torch.cuda.is_available() else "cpu"


def parseArguments(): 
    parser = argparse.ArgumentParser()

    # Necessary variables
    parser.add_argument("--embedder", type=str, required=True)
    parser.add_argument("--cls_type", type=str, default="linear")
    parser.add_argument("--is_challenge", action="store_true")
    parser.add_argument("--num_rows", type=int, default=-1,
                        help="Number of samples for finetuning. -1 means all samples.")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1123)
    parser.add_argument("--data_dir_path", type=str, default="./data/piqa")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report_step", type=int, default=100)
    
    # Necessary training parameters
    parser.add_argument("--input_size", type=int, default=768)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--train_step", type=int, default=1000)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulation_step", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()

    # I/O parameters
    dir_path, config_path, model_path, vocab_path = get_embedder_path(args.embedder)
    parser.add_argument("--dir_path", type=str, default=dir_path)
    parser.add_argument("--config_path", type=str, default=config_path)
    parser.add_argument("--model_path", type=str, default=model_path)
    parser.add_argument("--vocab_path", type=str, default=vocab_path)

    train_data_path, train_label_path = get_data_path(args.data_dir_path, split="train") 
    valid_data_path, valid_label_path = get_data_path(args.data_dir_path, split="valid")
    parser.add_argument("--train_data_path", type=str, default=train_data_path)
    parser.add_argument("--train_label_path", type=str, default=train_label_path)
    parser.add_argument("--valid_data_path", type=str, default=valid_data_path)
    parser.add_argument("--valid_label_path", type=str, default=valid_label_path)

    # Output parameter
    output_name = get_output_name(args)
    parser.add_argument("--output_name", type=str, default=output_name)
    args = parser.parse_args()

    return args

def get_output_name(args):
    output_name = "transformer_probing_{}_{}_R{}_S{}_B{}_LR{}".format(
        args.embedder, args.cls_type, args.num_runs, args.train_step, args.batch_size*args.accumulation_step, args.learning_rate
    )
    
    if args.is_challenge:
        output_name = "challenge_{}".format(output_name)

    return output_name

def get_logger(args):
    os.makedirs("./logs/piqa", exist_ok=True)
    os.makedirs("./logs/piqa/{}".format(args.output_name), exist_ok=True)
    logging.basicConfig(level=logging.INFO, \
            format = '%(asctime)s %(levelname)s: %(message)s', \
            datefmt = '%m/%d %H:%M:%S %p', \
            filename = './logs/piqa/{}/{}.log'.format(args.output_name, args.output_name), \
            filemode = 'w'
    )
    return logging.getLogger(__name__)

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

def output_predictions(true_list, best_pred_list, run_id, args):
    output_path = "./outputs/piqa/{}".format(args.output_name)
    logger.info("Output predictions to {}".format(output_path))
    os.makedirs(output_path, exist_ok=True)

    if not args.is_challenge:
        df = pd.DataFrame({"label":true_list, "pred":best_pred_list})
    else:
        true_list = true_list.reshape([-1,2])
        best_pred_list = best_pred_list.reshape([-1,2])
        df = pd.DataFrame({"label1": true_list[:,0], "label2": true_list[:,1], "pred1":best_pred_list[:,0], "pred2":best_pred_list[:,1]})
    df.to_csv("{}/predictions_{}.csv".format(output_path, run_id), index=False)

def train(model, optimizer, data, valid_id, args):
    model.train()
    loader = DataLoader(data, shuffle=True, pin_memory=True, batch_size=args.batch_size, drop_last=True)
    loader_iter = iter(loader)
    
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
            
def eval(model, data, valid_id, args):
    model.eval()
    true_list = []
    pred_list = []
    loader = DataLoader(data, shuffle=False, pin_memory=True, batch_size=args.batch_size)
    
    with torch.no_grad():
        for iter_id, batch in enumerate(loader):
            input_ids, attention_mask, token_type_ids, labels = tuple(t.to(device) for t in batch)
            logits, _ = model(input_ids, attention_mask, token_type_ids, is_challenge=args.is_challenge)
            
            logits = logits.detach().cpu()
            labels = labels.cpu()
            
            if not args.is_challenge:
                # Multiple choice
                preds = np.argmax(logits, axis=1)
                pred_list.append(preds)
                true_list.append(labels.numpy())
            else:
                # Binary classification
                labels_onehot = torch.Tensor(labels.size(0), input_ids.size(1)).cpu()
                labels_onehot.zero_()
                labels_onehot.scatter_(1, labels.view(-1, 1), 1)
                
                preds = (logits > 0).float()
                pred_list.append(preds.numpy())
                true_list.append(labels_onehot.numpy())
            
    pred_list = np.concatenate(pred_list)
    true_list = np.concatenate(true_list)
    
    accuracy = (pred_list == true_list).sum() / true_list.size
    
    return accuracy, pred_list, true_list

def main(args):
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case = 'uncased')

    logger.info("Load and preprocess data")
    train_samples = load_data(args.train_data_path, label_path=args.train_label_path, num_rows=args.num_rows)
    valid_samples = load_data(args.valid_data_path, label_path=args.valid_label_path)
    train_tokenized_samples = get_tokenized_samples(train_samples["text"], tokenizer, args.max_seq_len)
    valid_tokenized_samples = get_tokenized_samples(valid_samples["text"], tokenizer, args.max_seq_len)

    logger.info("Retrieve PIQADataset")
    train_dataset = PIQADataset(train_tokenized_samples, train_samples["label"])
    valid_dataset = PIQADataset(valid_tokenized_samples, valid_samples["label"])

    # Multiple probing runs
    accuracy_list = []

    for run_id in range(args.num_runs):
        logger.info("#"*70)
        logger.info("#"*70)

        # Load pretrained embedder & Freeze the last layer
        embedder, _ = load_embedder(args)
        for name, params in embedder.named_parameters():
            if not (name.startswith("encoder.layer.11") or name.startswith("pooler")):
                params.requires_grad = False

        # Create classifier
        model = MultipleChoiceClassifier(embedder, hidden_dropout=0)
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
            #train(model, optimizer, train_dataset, valid_id, args)
            model.train()
            for iter_id in range(args.train_step):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize loader_iter
                    loader_iter = iter(loader)
                    batch = next(loader_iter)
                   
                input_ids, attention_mask, token_type_ids, labels = tuple(t.to(device) for t in batch)
                logits, loss = model(input_ids, attention_mask, token_type_ids, labels=labels, is_challenge=args.is_challenge)
                loss = loss / args.accumulation_step
                loss.backward()

                if (iter_id+1) % args.accumulation_step == 0:
                    optimizer.step()
                    model.zero_grad()
                
                if args.verbose and iter_id % args.report_step == 0:
                    logger.info(f"| Validation {valid_id:6d} | iter {iter_id:8d} | train_loss {loss.item():8.5f} |")
                    
            
            # Early stopping and Evaluation
            train_accuracy, _, _ = eval(model, train_dataset, valid_id, args)
            accuracy, pred_list, true_list = eval(model, valid_dataset, valid_id, args)
            print(f"| Epoch {valid_id:6d} | eval_acc {accuracy:8.5f} | train_acc {train_accuracy:8.5f} |")
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
        total_epochs = total_steps / (len(train_dataset) / args.batch_size / args.accumulation_step)
        logger.info(f"Stopped training after {valid_id:d} validation checks")
        logger.info(f"Trained for {total_steps:d} steps (batches) or {total_epochs:f} epochs")
        logger.info(f"Best model found with accuracy {float(best_accuracy):8.5f} at Validation {best_valid_id}")

        output_predictions(true_list, best_pred_list, run_id, args)

    logger.info("#"*70)
    logger.info("#"*70)

    accuracy_list = np.array(accuracy_list)
    mean_accuracy = np.mean(accuracy_list)
    std_accuracy  = np.std(accuracy_list)
    accuracy_list = [str(round(acc, 4)) for acc in accuracy_list]
    str_accuracy_list = ", ".join(accuracy_list)
    logger.info(f"{args.num_runs:4d} runs accuracy list: [{accuracy_list}]")
    logger.info(f"Mean accuracy: {mean_accuracy:8.4f}")
    logger.info(f"Standard deviation accuracy: {std_accuracy:8.4f}")
    logger.info("Done!")

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)
    set_seed(args.seed)

    main(args)
