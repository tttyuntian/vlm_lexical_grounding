from transformers import BertModel, BertConfig, BertTokenizerFast


def load_embedder(args):
    config = BertConfig.from_json_file(args.config_path)
    model = BertModel.from_pretrained(args.model_path, config = config)
    tokenizer = BertTokenizerFast(args.vocab_path, do_lower_case = 'uncased')
    return model, tokenizer
