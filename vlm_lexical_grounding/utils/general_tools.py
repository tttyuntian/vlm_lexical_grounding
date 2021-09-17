import random
import os
import numpy as np
import torch
from transformers import BertModel, BertConfig, BertTokenizerFast

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

def get_embedder_path(embedder):
    if embedder == "BERT":
        dir_path = './models/uncased_L-12_H-768_A-12'
        config_path = '{}/bert_config.json'.format(dir_path)
        model_path = '{}/pytorch_model.bin'.format(dir_path)
        vocab_path = '{}/vocab.txt'.format(dir_path)
    elif embedder == "VideoBERT_randmask_text":
        dir_path = "./models/bert_ckpts_20210121"
        config_path = "{}/bert_config.json".format(dir_path)
        model_path = "{}/randmask_text/pytorch_model.bin".format(dir_path)
        vocab_path = "{}/vocab.txt".format(dir_path)
    elif embedder == "VideoBERT_randmask_vt":
        dir_path = "./models/bert_ckpts_20210121"
        config_path = "{}/bert_config.json".format(dir_path)
        model_path = "{}/randmask_vt/pytorch_model.bin".format(dir_path)
        vocab_path = "{}/vocab.txt".format(dir_path)
    elif embedder == "VideoBERT_topmask_text":
        dir_path = "./models/bert_ckpts_20210121"
        config_path = "{}/bert_config.json".format(dir_path)
        model_path = "{}/topmask_text/pytorch_model.bin".format(dir_path)
        vocab_path = "{}/vocab.txt".format(dir_path)
    elif embedder == "VideoBERT_topmask_vt":
        dir_path = "./models/bert_ckpts_20210121"
        config_path = "{}/bert_config.json".format(dir_path)
        model_path = "{}/topmask_vt/pytorch_model.bin".format(dir_path)
        vocab_path = "{}/vocab.txt".format(dir_path)
    elif embedder == "VisualBERT":
        dir_path = "./models/visualbert"
        config_path = "{}/bert_config.json".format(dir_path)
        model_path = "{}/visualbert_vt/pytorch_model.bin".format(dir_path)
        vocab_path = "{}/vocab.txt".format(dir_path)
    elif embedder == "VisualBERT_Text":
        dir_path = "./models/visualbert"
        config_path = "{}/bert_config.json".format(dir_path)
        model_path = "{}/visualbert_text/pytorch_model.bin".format(dir_path)
        vocab_path = "{}/vocab.txt".format(dir_path)
    else:
        assert False, "No model {} found.".format(embedder)
    return dir_path, config_path, model_path, vocab_path

