import os
import logging
import argparse

import pandas as pd
import numpy as np
import torch

from ..utils.general_tools import set_seed, get_embedder_path
from adj_noun_tools import load_embedder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def parseArguments():
    parser = argparse.ArgumentParser()

    # Necessary variables
    #parser.add_argument("--cluster_type", type=str, required=True,
    #                    help="The granularity of cluster, either target or bigram. target represents clusters of adj/nouns.")
    parser.add_argument("--bigram_occur_threshold", type=int, default=10)
    #parser.add_argument("--retrieve_threshold", type=int, default=-1,
    #                    help="Maximum number of retrieved embeddings for a bigram. -1 means retriving every embedding.")
    parser.add_argument("--unique_bigram_threshold", type=int, default=5,
                        help="Minimum number of unique bigrams for a targetw word.")
    #parser.add_argument("--POS", type=str, required=True)
    parser.add_argument("--embedder", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=1123)
    args = parser.parse_args()

    # Model I/O parameters
    dir_path, config_path, model_path, vocab_path = get_embedder_path(args.embedder)
    parser.add_argument("--dir_path", type=str, default=dir_path)
    parser.add_argument("--config_path", type=str, default=config_path)
    parser.add_argument("--model_path", type=str, default=model_path)
    parser.add_argument("--vocab_path", type=str, default=vocab_path)
    args = parser.parse_args()

    # I/O parameters
    output_name, output_dir = get_output_path(args)
    parser.add_argument("--output_name", type=str, default=output_name)
    parser.add_argument("--output_dir", type=str, default=output_dir)
    parser.add_argument("--data_path", type=str, default="./data/wikiHow/wikihowAll_clean_single.csv")
    parser.add_argument("--general_statistics_path", type=str, default="./outputs/general_statistics/general_statistics_Rall/df_general.csv")
    parser.add_argument("--mit_states_path", type=str, default="./data/mit_states/release_dataset/adj_ants.csv")
    
    # Model parameters
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    return args

def get_logger(args):
    os.makedirs("./logs/get_target_embs", exist_ok=True)
    os.makedirs("./logs/get_target_embs/{}".format(args.output_name), exist_ok=True)
    logging.basicConfig(
        level = logging.DEBUG, 
        format = '%(asctime)s %(levelname)s: %(message)s', 
        datefmt = '%m/%d %H:%M:%S %p', 
        filename = './logs/get_target_embs/{}/{}.log'.format(args.output_name, args.output_name),
        filemode = 'w'
    )
    return logging.getLogger()

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

def get_output_path(args):
    output_name = "{}_B{}_U{}".format(
        args.embedder, 
        args.bigram_occur_threshold,
        args.unique_bigram_threshold,
        #args.retrieve_threshold,
    )
    output_dir = "./outputs/get_target_embs/{}".format(output_name)

    os.makedirs("./outputs/get_target_embs", exist_ok=True)
    os.makedirs("./outputs/get_target_embs/{}".format(output_name), exist_ok=True)
    return output_name, output_dir

def get_target_df(df, adj_ant_dict, args):
    # Bigram occurrence filter
    target_df = df[df.bigram_occur>=args.bigram_occur_threshold]

    # POS filter
    target_df = target_df[target_df.POS=="ADJ"]

    # string filter for both target and bigram
    condition1 = target_df.target.str.isalpha()
    condition2 = target_df.bigram.str.isalpha()
    target_df = target_df[condition1 & condition2]

    # MIT states adjectives filter
    target_df = target_df[target_df.target.isin(adj_ant_dict.keys())]

    pass_unique_threshold_list = (target_df.groupby(["bigram"]).size()>=args.unique_bigram_threshold).to_dict()
    target_df.loc[:, "pass_unique_threshold"] = False
    for row_id in target_df.index:
        noun = target_df.bigram[row_id]
        if (noun in pass_unique_threshold_list) and pass_unique_threshold_list[noun]:
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

def get_spans(tokenized_outputs, adjective_id, noun_id, bigram_id):
    """Retrieve adjective/noun/bigram spans in the input sequence.
    
    Args:
        tokenized_outputs (BatchEncoding): Tokenized outputs from BertTokenizer.
        adjective_id (list): Word indices of adjective.
        noun_id (list): Word indices of noun.
        bigram_id (list): Word indices of bigram.
    
    Returns:
        adjective_spans (list): A list of int which defines the adjective spans in the input sequence. 
        noun_spans (list): A list of int which defines the noun spans in the input sequence. 
        bigram_spans (list): A list of int which defines the bigram spans in the input sequence. 
    """ 
    adjective_spans = []
    noun_spans = []
    bigram_spans = []

    for bigram_start in range(len(tokenized_outputs.input_ids)-len(bigram_id)+1):
        # Iterate through input_ids to find the bigram spans
        bigram_end = bigram_start + len(bigram_id)
        bigram_window = tokenized_outputs.input_ids[bigram_start:bigram_end]

        if bigram_id == bigram_window:
            # Divide this bigram span into adjective span and noun span
            bigram_span = list(range(bigram_start, bigram_end))
            adjective_spans.append(bigram_span[:len(adjective_id)])
            noun_spans.append(bigram_span[len(adjective_id):])
            bigram_spans.append(bigram_span)
    
    return adjective_spans, noun_spans, bigram_spans

def get_filter_regex(df):
    filter_regex = ""
    for target in df.target.unique():
        filter_regex += r"\b{}\b|".format(target)
    filter_regex = filter_regex.strip("|")
    return filter_regex

def main(args):
    logger.info("Loading data and general statistics.")
    raw_data = pd.read_csv(args.data_path)
    adj_ant_dict = get_mit_states_data(args)    
    df = pd.read_csv(args.general_statistics_path)
    df = get_target_df(df, adj_ant_dict, args)

    logger.info("Load embedder and tokenizer.")
    embedder, tokenizer = load_embedder(args)
    embedder.to(device)
    embedder.eval()

    logger.info("Target words filter.")
    filter_regex = get_filter_regex(df)
    target_data_ids = raw_data.text.str.contains(filter_regex, case=False, regex=True)
    target_data = raw_data[target_data_ids].reset_index().copy()

    logger.info("Start computing the target embeddings.")
    adj_emb_list, noun_emb_list, bigram_emb_list = [], [], []
    adj_list = []
    noun_list = []
    data_row_id_list = []
    text_list = []
    count_list = []

    for target_row_id in range(len(df)):
    #for target_row_id in range(5):
        count = 0
        target = df.target[target_row_id]
        bigram = df.bigram[target_row_id]
        
        #if args.POS=="ADJ":
        adjective, noun = target, bigram
        bigram = "{} {}".format(target, bigram)
        #else:
        #    adjective, noun = bigram, target
        #    bigram = "{} {}".format(bigram, target)
        
        # Target word filter
        filter_regex = r"\b{}\b".format(bigram)
        data_ids = target_data.text.str.contains(filter_regex, case=False, regex=True)
        data = target_data[data_ids].reset_index().copy()
        
        # Retrieve target and bigram ids
        adjective_id = tokenizer(adjective, add_special_tokens=False).input_ids
        noun_id = tokenizer(noun, add_special_tokens=False).input_ids
        bigram_id = tokenizer(bigram, add_special_tokens=False).input_ids
        
        for data_row_id in range(len(data)):
        #for data_row_id in range(5):
        
            # If collected target embeddings == threshold, then skip this target
            #if (args.retrieve_threshold > 0) and (count == args.retrieve_threshold):
            #    break
            
            # Retrieve target spans
            text = data.text[data_row_id]
            tokenized_outputs = tokenizer(
                text, 
                add_special_tokens=False,
                truncation=True,
                max_length = args.max_length,
            )
            spans = get_spans(tokenized_outputs, adjective_id, noun_id, bigram_id)
            
            # Retrieve word representations
            tokenized_results = {k:torch.tensor(v).long().unsqueeze(0).to(device) for k,v in tokenized_outputs.items()}
            with torch.no_grad():
                outputs = embedder(**tokenized_results)
            
            for adj_span, noun_span, bigram_span in zip(spans[0], spans[1], spans[2]):
                bigram_emb = outputs[0].squeeze()[bigram_span].detach().cpu().numpy()
                count += 1

                adj_emb_list.append(np.mean(bigram_emb[:len(adjective_id)], axis=0))
                noun_emb_list.append(np.mean(bigram_emb[len(adjective_id):], axis=0))
                bigram_emb_list.append(np.mean(bigram_emb, axis=0))
                adj_list.append(adjective)
                noun_list.append(noun)
                data_row_id_list.append(data["index"][data_row_id])
                text_list.append(text)
                
                # If collected target embeddings == threshold, then skip this target
                #if (args.retrieve_threshold > 0) and (count == args.retrieve_threshold):
                #    break
        
        # Keep track of count for each bigram
        count_list.extend([count]*count)

        logger.info(f"Progress: {target_row_id}/{len(df)}")
        """
        if (args.retrieve_threshold > 0) and (count != args.retrieve_threshold):
            logger.info("="*50)
            logger.info(f"Target <{target}> in Bigram <{bigram}> occur {count} times.")
            logger.info("="*50)
        """

    # Generate df for general statistics
    adj_emb_list = np.array(adj_emb_list)
    noun_emb_list = np.array(noun_emb_list)
    bigram_emb_list = np.array(bigram_emb_list)

    df_info = pd.DataFrame(
        {
            "adjective": np.array(adj_list),
            "noun": np.array(noun_list),
            "row_id": np.array(data_row_id_list),
            "text": np.array(text_list),
            "count": np.array(count_list),
        }
    )

    logger.info("Output dataframe to {}".format(args.output_dir))
    df_info.to_csv("{}/df_info.csv".format(args.output_dir), index=False)
    np.save("{}/adj_embeddings.npy".format(args.output_dir), adj_emb_list)
    np.save("{}/noun_embeddings.npy".format(args.output_dir), noun_emb_list)
    np.save("{}/bigram_embeddings.npy".format(args.output_dir), bigram_emb_list)

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)
    logger.info("Device: {}".format(device))
    
    main(args)

