import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class PIQADataset(Dataset):
    def __init__(self, tokenized_samples, labels=None):
        self.input_ids = tokenized_samples.input_ids
        self.attention_mask = tokenized_samples.attention_mask
        self.token_type_ids = tokenized_samples.token_type_ids
        self.labels = labels if labels is not None else None
    
    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        input_ids = torch.from_numpy(self.input_ids[idx])
        attention_mask = torch.from_numpy(self.attention_mask[idx])
        token_type_ids = torch.from_numpy(self.token_type_ids[idx])
        if self.labels is None:
            return input_ids, attention_mask, token_type_ids
        else:
            label = torch.tensor(self.labels[idx]).long()
            return input_ids, attention_mask, token_type_ids, label

class PIQAEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels=None):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return self.embeddings.shape[0]
    
    def __getitem__(self, idx):
        embeddings = torch.from_numpy(self.embeddings[idx])
        if self.labels is None:
            return embeddings
        else:
            label = torch.tensor(self.labels[idx]).long()
            return embeddings, label

def load_data(data_path, label_path=None, num_rows=None):
    samples = defaultdict(list)
    with open(data_path, "r") as f:
        json_list = list(f)
    for line in json_list:
        line = json.loads(line)
        samples["text"].append("{} [SEP] {}".format(line["goal"], line["sol1"]))
        samples["text"].append("{} [SEP] {}".format(line["goal"], line["sol2"]))
    if label_path is not None:
        with open(label_path, "r") as f:
            samples["label"] = f.readlines()
            samples["label"] = [int(label) for label in samples["label"]]
    if (num_rows is not None) and (num_rows != -1):
        samples["text"] = samples["text"][:num_rows*2]
        if label_path is not None:
            samples["label"] = samples["label"][:num_rows]
    return samples

def load_emb_data(data_path, label_path, hidden_size, num_rows=None):
    samples = defaultdict()
    samples["embedding"] = np.load(data_path)
    samples["embedding"] = samples["embedding"].reshape((-1, 2, hidden_size))
    with open(label_path, "r") as f:
        samples["label"] = f.readlines()
        samples["label"] = [int(label) for label in samples["label"]]
    
    if (num_rows is not None) and (num_rows != -1):
        samples["embedding"] = samples["embedding"][:num_rows]
        samples["label"] = samples["label"][:num_rows]
    return samples

def get_tokenized_samples(samples, tokenizer, max_length):
    results = tokenizer(samples, truncation=True, max_length=max_length, padding="max_length")
    for k, v in results.items():
        results[k] = np.array(v).reshape((-1, 2, max_length))
    return results
