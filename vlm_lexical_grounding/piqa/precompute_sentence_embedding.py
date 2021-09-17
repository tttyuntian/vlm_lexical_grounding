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

from ..utils.simple_classifier import MultipleChoiceClassifier
from ..utils.general_tools import set_seed, get_embedder_path
from .piqa_tools import get_data_path, load_embedder
from .piqa_data import PIQADataset, get_tokenized_samples, load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parseArguments(): 
    parser = argparse.ArgumentParser()

    # Necessary variables
    parser.add_argument("--embedder", type=str, required=True)
    parser.add_argument("--num_rows", type=int, default=-1,
                        help="Number of samples for finetuning. -1 means all samples.")
    parser.add_argument("--seed", type=int, default=1123)
    parser.add_argument("--data_dir_path", type=str, default="./data/piqa")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report_step", type=int, default=1000)
    args = parser.parse_args()

    # I/O parameters
    dir_path, config_path, model_path, vocab_path = get_embedder_path(args.embedder)
    parser.add_argument("--dir_path", type=str, default=dir_path)
    parser.add_argument("--config_path", type=str, default=config_path)
    parser.add_argument("--model_path", type=str, default=model_path)
    parser.add_argument("--vocab_path", type=str, default=vocab_path)

    # Output parameter
    parser.add_argument("--output_name", type=str, default="{}_precompute".format(args.embedder))
    args = parser.parse_args()

    return args

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

def precompute_sentence_embedding(model, data, args):
    embeddings = []
    loader = DataLoader(data, shuffle=False, pin_memory=True, batch_size=args.batch_size)

    with torch.no_grad():
        for iter_id, batch in enumerate(loader):
            input_ids, attention_mask, token_type_ids = tuple(t.to(device) for t in batch)

            input_ids = input_ids.view(-1, input_ids.size(-1))
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            pooled_output = outputs[1].detach().cpu().numpy()

            embeddings.append(pooled_output)
            if args.verbose and iter_id % args.report_step == 0:
                logger.info(f"| iter_id {iter_id:8d} |")
    
    embeddings = np.concatenate(embeddings)
    return embeddings

def output_embedding(embeddings, split, args):
    os.makedirs("{}/precompute_embs".format(args.data_dir_path), exist_ok=True)
    output_path = "{}/precompute_embs/{}_{}.npy".format(args.data_dir_path, args.embedder, split)
    np.save(output_path, embeddings)
    
    logger.info("*"*70)
    logger.info(f"Save {split} embeddings from {args.embedder} to {output_path}")
    logger.info("*"*70)

def main(args):
    logger.info("Load pretrained embedder and tokenizer")
    embedder, tokenizer = load_embedder(args)
    embedder.to(device)
    embedder.eval()

    #for split in ["train", "valid", "test"]:
    for split in ["train", "valid"]:
        logger.info("="*70)
        logger.info(f"Start computing for split: {split}")
        
        # Preprocess data
        data_path, _ = get_data_path(args.data_dir_path, split=split)
        samples = load_data(data_path, num_rows=args.num_rows)
        tokenized_samples = get_tokenized_samples(samples["text"], tokenizer, embedder.config.max_position_embeddings)
        dataset = PIQADataset(tokenized_samples)

        # Compute sentence embeddings
        embeddings = precompute_sentence_embedding(embedder, dataset, args)

        # Output computed embeddings
        output_embedding(embeddings, split, args)

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)
    set_seed(args.seed)

    main(args)
