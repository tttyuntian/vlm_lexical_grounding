# Vision-and-Language Pretraining & Lexical Grounding
PyTorch code for the Findings of EMNLP 2021 paper "Does Vision-and-Language Pretraining Improve Lexical Grounding?" (Tian Yun, Chen Sun, and Ellie Pavlick).

**Outline**
(Placeholder)

## Installation
```shell script
pip install -r requirements.txt
```

Require python 3.6+ (to support huggingface [transformers](https://github.com/huggingface/transformers)).

## Physical Commonsense QA

### Download Data
Download [PIQA](https://yonatanbisk.com/piqa/):
  ```shell script
  mkdir -p data/piqa
  wget https://yonatanbisk.com/piqa/data/train.jsonl -P data/piqa
  wget https://yonatanbisk.com/piqa/data/train-labels.lst -P data/piqa
  wget https://yonatanbisk.com/piqa/data/valid.jsonl -P data/piqa
  wget https://yonatanbisk.com/piqa/data/valid-labels.lst -P data/piqa
  wget https://yonatanbisk.com/piqa/data/tests.jsonl -P data/piqa
  ```
  
### Precompute Sentence Embeddings
  ```shell script
  # Available `embedder` are:
  # - BERT
  # - VideoBERT_randmask_text
  # - VideoBERT_randmask_vt
  # - VideoBERT_topmask_text
  # - VideoBERT_topmask_vt
  # - VisualBERT_text
  # - VisualBERT_vt
  bash scripts/piqa/precompute_sentence_embedding.py -e [embedder]
  ```

### Probing Experiments
  ```shell script
  # For probing experiments of "linear/MLP" probes in the paper
  # `cls_type` can either be `linear` or `mlp`.
  bash scripts/piqa/piqa_probing.py -e [embedder] -c [cls_type]
  
  # For probing experiment of "transformer" probe in the paper
  bash scripts/piqa/piqa_transformer_probing.py -e [embedder]
  ```
  The default arguments of both commands will run the experiments for 5 times and log the averaged performance metric. You can modify the `num_runs` in the scripts to control the number of runs. The logs will be written to `logs/piqa/`, while the outputs (i.e. predictions on validation set of PIQA) will be written to `outputs/piqa/`.
  
  

