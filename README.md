# Vision-and-Language Pretraining & Lexical Grounding
  PyTorch code for the Findings of EMNLP 2021 paper "Does Vision-and-Language Pretraining Improve Lexical Grounding?" (Tian Yun, Chen Sun, and Ellie Pavlick). [PDF](https://arxiv.org/abs/2109.10246)

  If you find this project useful, please cite our paper:
  ```
  @misc{yun2021does,
        title={Does Vision-and-Language Pretraining Improve Lexical Grounding?}, 
        author={Tian Yun and Chen Sun and Ellie Pavlick},
        year={2021},
        eprint={2109.10246},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
  }
  ```

**Outline**
* [1. Physical Commonsense QA](#1-physical-commonsense-qa)
  * [1.1. Download Data](#11-download-data)
  * [1.2. Precompute Sentence Embeddings](#12-precompute-sentence-embeddings)
  * [1.3. Probing Experiments](#13-probing-experiments)
* [2. Adjective Noun Composition](#2-adjective-noun-composition)
  * [2.1. Download and Preprocess Data](#21-download-and-preprocess-data)
  * [2.2. Find Adjective Noun Candidate Pairs and Precompute Noun Embeddings](#22-find-adjective-noun-candidate-pairs-and-precompute-noun-embeddings)
  * [2.3. K-Means Clustering](#23-k-means-clustering)
  * [2.4. Adjective Probing](#24-adjective-probing)
* [3. Pretrained VL and Text-only Models](#3-pretrained-vl-and-text-only-models)

## Installation
  ```shell script
  pip install -r requirements.txt
  ```
  Require python 3.6+ (to support huggingface [transformers](https://github.com/huggingface/transformers)).

## 1. Physical Commonsense QA
  In this section (corresponding to Section 4.1 of the [paper](https://arxiv.org/pdf/2109.10246.pdf)), we want to explore if VL pretraining yields gains to an extrinsic task that doesn't explicitly require representing non-text inputs but intuitively requires physical commonsense knowledge.

### 1.1. Download Data
  Download [PIQA](https://yonatanbisk.com/piqa/):
  ```shell script
  mkdir -p data/piqa
  wget https://yonatanbisk.com/piqa/data/train.jsonl -P data/piqa
  wget https://yonatanbisk.com/piqa/data/train-labels.lst -P data/piqa
  wget https://yonatanbisk.com/piqa/data/valid.jsonl -P data/piqa
  wget https://yonatanbisk.com/piqa/data/valid-labels.lst -P data/piqa
  wget https://yonatanbisk.com/piqa/data/tests.jsonl -P data/piqa
  ```
  
### 1.2. Precompute Sentence Embeddings
  We precompute the sentence embeddings to boost up the probing experiments.
  ```shell script
  # Available `embedder` are:
  # - BERT
  # - VideoBERT_randmask_text
  # - VideoBERT_randmask_vt
  # - VideoBERT_topmask_text
  # - VideoBERT_topmask_vt
  # - VisualBERT_text
  # - VisualBERT_vt
  bash scripts/piqa/precompute_sentence_embedding.sh -e [embedder]
  ```

### 1.3. Probing Experiments
  We measure the quality of the representations with 3 different probing heads: **Linear probe**, **MLP probe**, and a **Transformer probe**. **Transformer probe** is to finetune the last transformer encoder layer and a linear layer on top of it. 
  ```shell script
  # For probing experiments of "linear/MLP" probes in the paper
  # `cls_type` can either be `linear` or `mlp`.
  bash scripts/piqa/piqa_probing.sh -e [embedder] -c [cls_type]
  
  # For probing experiment of "transformer" probe in the paper
  bash scripts/piqa/piqa_transformer_probing.sh -e [embedder]
  ```
  The default arguments of both commands will run the experiments for 5 times and log the averaged performance metric. You can modify the `num_runs` in the scripts to control the number of runs. The logs will be written to `logs/piqa/`, while the outputs (i.e. predictions on validation set of PIQA) will be written to `outputs/piqa/`.


## 2. Adjective Noun Composition
  This section corresponds to Section 4.3 of the [paper](https://arxiv.org/pdf/2109.10246.pdf). We explore whether multimodal pretraining impacts conceptual structure at the lexical level. To look into this, we focus on adjective-noun composition which provides a simple way of defining a space of visually groundable objects and properties that we expect conceptual representations to encode.

### 2.1. Download and Preprocess Data
  We pick [WikiHow](https://github.com/mahnazkoupaee/WikiHow-Dataset), a dataset about step-by-step instructions of daily tasks. We first split the instructions into single sentences, and then run a bigram search over all the sentences to extract adjective-noun pairs.

  We also use the "visually-groundable" adjectives in [MIT States data](http://web.mit.edu/phillipi/Public/states_and_transformations/index.html) as our adjective filter. 

  1. Download [WikiHow](https://github.com/mahnazkoupaee/WikiHow-Dataset) from this [link](https://ucsb.app.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358):
      ```shell script
      # Download WikiHow
      mkdir -p data/wikiHow
      mv wikihowAll.csv data/wikiHow

      # Preprocess WikiHow
      python3 -m vlm_lexical_grounding.adj_noun_composition.wikihow_preprocess
      ```
  2. Download [MIT States data](http://web.mit.edu/phillipi/Public/states_and_transformations/index.html):
      ```shell script
      mkdir -p data/mit_states
      wget http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip -P data/mit_states
      unzip release_dataset.zip
      ```

### 2.2. Find Adjective Noun Candidate Pairs and Precompute Noun Embeddings
  We will find the pairs of an adjective and a noun, and then precompute the noun representations for K-Means clustering and adjective probing experiments. This step is necessary before we proceed to the two experiments.
  ```shell script
  # Find `adjective noun` candidate pairs
  bash scripts/adj_noun_composition/general_statistics.sh
  
  # Precompute noun embeddings
  # Available `embedder` are:
  # - BERT
  # - VideoBERT_randmask_text
  # - VideoBERT_randmask_vt
  # - VideoBERT_topmask_text
  # - VideoBERT_topmask_vt
  # - VisualBERT_text
  # - VisualBERT_vt
  bash scripts/adj_noun_composition/get_target_embs.sh -e [embedder]
  ```

### 2.3. K-Means Clustering
  We use K-Means to cluster the representations of each noun, with *K* equals to the number of unique adjectives that modifies the noun in our dataset.
  ```shell script
  bash scripts/adj_noun_composition/kmeans_clustering.sh -e [embedder]
  ```
  
### 2.4. Adjective Probing
  We attempt to evaluate the adjective information that is linearly encoded in the noun representations.
  ```shell script
  bash scripts/adj_noun_composition/adjective_probing.sh -e [embedder]
  ```
  
## 3. Pretrained VL and Text-only Models
  - [BERT](https://drive.google.com/file/d/1mLJsaVBa0yWPrAXUB2b102dK-fuloG_I/view?usp=sharing)
  - [VideoBERT](https://drive.google.com/file/d/1lYoEPlhtDwk32Lpje_98IjQAR3BJ5uCv/view?usp=sharing)
  - [VisualBERT](https://drive.google.com/file/d/1E5shEC54fLJImkyfchwuxNFnbMVBCs1j/view?usp=sharing)

  After downloading the zip files, move them to `models/` and `unzip *.zip`.
  ```shell script
  mv *.zip models/
  unzip *.zip
  ```

## 4. Acknowledgements
  We thank the reviewers and [Liunian (Harold) Li](https://liunian-harold-li.github.io/) for their helpful discussions. Part of the code are built based on huggingface [transformers](https://github.com/huggingface/transformers).

