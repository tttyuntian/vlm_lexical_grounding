import os
import logging
import argparse
from collections import defaultdict

import pandas as pd
import numpy as np
import spacy
import torch

from ..utils.general_tools import set_seed, get_embedder_path

nlp = spacy.load("en_core_web_sm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def parseArguments():
    parser = argparse.ArgumentParser()

    # Necessary variables
    parser.add_argument("--num_rows", type=int, required=True,
                        help="Number of rows from dataset. -1 means all samples.")
    parser.add_argument("--seed", type=int, default=1123)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report_step", type=int, default=5000)
    args = parser.parse_args()

    # I/O parameters
    output_name, output_dir = get_output_path(args)
    parser.add_argument("--output_name", type=str, default=output_name)
    parser.add_argument("--output_dir", type=str, default=output_dir)
    parser.add_argument("--data_path", type=str, default="./data/wikiHow/wikihowAll_clean_single.csv")
    args = parser.parse_args()

    return args

def get_logger(args):
    os.makedirs("./logs/general_statistics", exist_ok=True)
    os.makedirs("./logs/general_statistics/{}".format(args.output_name), exist_ok=True)
    logging.basicConfig(
        level = logging.DEBUG, 
        format = '%(asctime)s %(levelname)s: %(message)s', 
        datefmt = '%m/%d %H:%M:%S %p', 
        filename = './logs/general_statistics/{}/{}.log'.format(args.output_name, args.output_name),
        filemode = 'w'
    )
    return logging.getLogger()

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

def get_output_path(args):
    num_rows = "all" if args.num_rows == -1 else args.num_rows
    output_name = "general_statistics_R{}".format(num_rows)
    output_dir = "./outputs/general_statistics/{}".format(output_name)
    os.makedirs("./outputs/general_statistics", exist_ok=True)
    os.makedirs("./outputs/general_statistics/{}".format(output_name), exist_ok=True)
    return output_name, output_dir

def get_df_general(df_noun, df_adj, adj_for_noun_counter, noun_for_adj_counter):
    """Convert dataframes into a large dataframe including all statistics."""
    target_list = []
    target_pos_list = []
    target_count_list = []
    bigram_list = []
    bigram_count_list = []

    # Append every NOUN targets
    for k, v in adj_for_noun_counter.items():
        for k_sub, v_sub in v.items():
            target_list.append(k)
            target_pos_list.append("NOUN")
            target_count_list.append(df_noun[df_noun.target==k].total_count.tolist()[0])
            bigram_list.append(k_sub)
            bigram_count_list.append(v_sub)
    
    # Append every ADJ targets
    for k, v in noun_for_adj_counter.items():
        for k_sub, v_sub in v.items():
            target_list.append(k)
            target_pos_list.append("ADJ")
            target_count_list.append(df_adj[df_adj.target==k].total_count.tolist()[0])
            bigram_list.append(k_sub)
            bigram_count_list.append(v_sub)
            
    return pd.DataFrame(
        {
            "target": target_list,
            "POS": target_pos_list,
            "target_occur": target_count_list,
            "bigram": bigram_list,
            "bigram_occur": bigram_count_list,
        }
    )

def main(args):
    logger.info("Loading data and word frequency data.")
    raw_data = pd.read_csv(args.data_path)

    logger.info("Start computing the statistics.")
    adj_counter = defaultdict(int)
    noun_counter = defaultdict(int)
    adj_for_noun_counter = {}
    noun_for_adj_counter = {}
    adj_for_noun_bigram_cnt = 0
    noun_for_adj_bigram_cnt = 0
    
    num_rows = args.num_rows if args.num_rows > 0 else len(raw_data)

    for row_id in range(num_rows):
        # Bigram search using nltk parser
        text = raw_data.text[row_id]
        doc = nlp(text)

        for i, token in enumerate(doc):
            if token.pos_ == "NOUN":
                lower_token = token.text.lower()
                noun_counter[lower_token] += 1
                if i > 0 and doc[i-1].pos_=="ADJ":
                    # if the current token is NOUN and the previous token is ADJ, then append this bigram
                    lower_prev = doc[i-1].text.lower()
                    if lower_token not in adj_for_noun_counter:
                        adj_for_noun_counter[lower_token] = defaultdict(int)
                    adj_for_noun_counter[lower_token][lower_prev] += 1
                    adj_for_noun_bigram_cnt += 1

            elif token.pos_ == "ADJ":
                lower_token = token.text.lower()
                adj_counter[lower_token] += 1
                if i < len(doc) and doc[i+1].pos_=="NOUN":
                    # if current token is ADJ and the next token is NOUN, then append this bigram
                    lower_next = doc[i+1].text.lower()
                    if lower_token not in noun_for_adj_counter:
                        noun_for_adj_counter[lower_token] = defaultdict(int)
                    noun_for_adj_counter[lower_token][lower_next] += 1
                    noun_for_adj_bigram_cnt += 1

        if args.verbose and row_id % args.report_step == 0:
            logger.info("Progress: {}/{}".format(row_id, args.num_rows))

    logger.info(f"bigram count match: {adj_for_noun_bigram_cnt==noun_for_adj_bigram_cnt}")
    logger.info(f"bigram count: {adj_for_noun_bigram_cnt}")
    
    # Generate df for general statistics
    df_adj = pd.DataFrame(
        {
            "target": list(dict(adj_counter).keys()),
            "total_count": list(dict(adj_counter).values()),
        }
    )
    df_noun = pd.DataFrame(
        {
            "target": list(dict(noun_counter).keys()),
            "total_count": list(dict(noun_counter).values()),
        }
    )
    df_general = get_df_general(df_noun, df_adj, adj_for_noun_counter, noun_for_adj_counter)

    logger.info("Output dataframe to {}".format(args.output_dir))
    df_adj.to_csv("{}/df_adj.csv".format(args.output_dir), index=False)
    df_noun.to_csv("{}/df_noun.csv".format(args.output_dir), index=False)
    df_general.to_csv("{}/df_general.csv".format(args.output_dir), index=False)

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)
    logger.info("Device: {}".format(device))
    
    main(args)

