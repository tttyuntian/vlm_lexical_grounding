from transformers import BertModel, BertConfig, BertTokenizer


def load_embedder(args):
    config = BertConfig.from_json_file(args.config_path)
    model = BertModel.from_pretrained(args.model_path, config = config)
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case = 'uncased')
    return model, tokenizer

def get_data_path(data_dir_path, split):
    data_path  = "{}/{}.jsonl".format(data_dir_path, split)
    if split in ["train", "valid"]:
        label_path = "{}/{}-labels.lst".format(data_dir_path, split) 
    elif split in ["test"]:
        label_path = None
    return data_path, label_path

def get_emb_data_path(args, split):
    data_path = "{}/precompute_embs/{}_{}.npy".format(args.data_dir_path, args.embedder, split)
    if split in ["train", "valid"]:
        label_path = "{}/{}-labels.lst".format(args.data_dir_path, split)
    elif split in ["test"]:
        label_path = None
    return data_path, label_path
