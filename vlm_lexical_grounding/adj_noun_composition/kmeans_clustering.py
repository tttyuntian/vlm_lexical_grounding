import os
import logging
import argparse

import pandas as pd
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def parseArguments():
    parser = argparse.ArgumentParser()

    # Necessary variables
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--cluster_type", type=str, required=True,
                        help="The type of cluster, either adjective or noun.")
    parser.add_argument("--method", type=str, default="kmeans")
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--bigram_occur_threshold", type=int, default=10)
    parser.add_argument("--unique_bigram_threshold", type=int, default=5)
    parser.add_argument("--sample_threshold", type=int, default=20)
    parser.add_argument("--embedder", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=1123)
    args = parser.parse_args()

    # I/O parameters
    output_name, output_dir = get_output_path(args)
    parser.add_argument("--output_name", type=str, default=output_name)
    parser.add_argument("--output_dir", type=str, default=output_dir)
    args = parser.parse_args()

    df_info_path, embeddings_path = get_data_path(args)
    parser.add_argument("--df_info_path", type=str, default=df_info_path)
    parser.add_argument("--embeddings_path", type=str, default=embeddings_path)
    parser.add_argument("--general_statistics_path", type=str, default="./outputs/general_statistics/general_statistics_Rall/df_general.csv")
    parser.add_argument("--mit_states_path", type=str, default="./data/mit_states/release_dataset/adj_ants.csv")
    args = parser.parse_args()

    return args

def get_logger(args):
    os.makedirs("./logs/clustering_final", exist_ok=True)
    os.makedirs("./logs/clustering_final/{}".format(args.output_name), exist_ok=True)
    logging.basicConfig(
        level = logging.DEBUG, 
        format = '%(asctime)s %(levelname)s: %(message)s', 
        datefmt = '%m/%d %H:%M:%S %p', 
        filename = './logs/clustering_final/{}/{}.log'.format(args.output_name, args.output_name),
        filemode = 'w'
    )
    return logging.getLogger()

def logging_args(args):
    for arg, value in vars(args).items():
        logger.info("Argument {}: {}".format(arg, value))

def get_output_path(args):
    output_name = "{}_{}_{}_B{}_U{}_S{}".format(
        args.method,
        args.cluster_type,
        args.embedder, 
        args.bigram_occur_threshold,
        args.unique_bigram_threshold,
        args.sample_threshold,
    )
    output_dir = "./outputs/clustering_final/{}".format(output_name)

    os.makedirs("./outputs/clustering_final", exist_ok=True)
    os.makedirs("./outputs/clustering_final/{}".format(output_name), exist_ok=True)
    return output_name, output_dir

def get_clean_df_info(df_info, target_df, args, seed):
    # Sample embeddings from every cluster (i.e. target/bigram)      
    target_col = "noun" if args.cluster_type=="NOUN" else "adjective"
    other_col = "noun" if target_col=="adjective" else "adjective"

    unique_target_list = target_df.target.unique()
    df_list = []

    for target in unique_target_list:
        # Retrieve df of this target
        curr_df = df_info[df_info[target_col]==target]

        # Sample the embeddings from all the target embeddings of this target
        sample_list = curr_df[other_col].value_counts().to_dict()
        for sample_target, sample_target_count in sample_list.items():
            sample_target_df = curr_df[curr_df[other_col]==sample_target]
            if sample_target_count > args.sample_threshold:
                sample_target_df = sample_target_df.sample(args.sample_threshold, random_state=seed)
            df_list.append(sample_target_df)

    final_df_info = pd.concat(df_list)
    logger.info(f"final_df_info number of rows: {len(final_df_info)}")

    return final_df_info

def get_data_path(args):
    name = "{}_B{}_U{}".format(
        args.embedder, 
        args.bigram_occur_threshold,
        args.unique_bigram_threshold,
    )
    if args.cluster_type=="ADJ":
        emb_name = "adj"
    elif args.cluster_type=="NOUN":
        emb_name = "noun"
    df_info_path = "./outputs/get_target_embs/{}/df_info.csv".format(name)
    embeddings_path = "./outputs/get_target_embs/{}/{}_embeddings.npy".format(name, emb_name)
    return df_info_path, embeddings_path

def get_target_df(df, adj_ant_dict, args):
    # Bigram occurrence filter
    target_df = df[df.bigram_occur>=args.bigram_occur_threshold]

    # POS filter
    pos_filter = "NOUN" if args.cluster_type=="NOUN" else "ADJ"
    target_df = target_df[target_df.POS==pos_filter]
    
    # string filter for both target and bigram
    condition1 = target_df.target.str.isalpha()
    condition2 = target_df.bigram.str.isalpha()
    target_df = target_df[condition1 & condition2]

    # MIT states adjectives filter
    if args.cluster_type == "ADJ":
        target_df = target_df[target_df.target.isin(adj_ant_dict.keys())]
    elif args.cluster_type == "NOUN":
        target_df = target_df[target_df.bigram.isin(adj_ant_dict.keys())]

    # Unique bigram filter
    pass_unique_threshold_list = (target_df.groupby(["target"]).size()>=args.unique_bigram_threshold).to_dict()
    target_df.loc[:, "pass_unique_threshold"] = False
    for row_id in target_df.index:
        target = target_df.target[row_id]
        if (target in pass_unique_threshold_list) and pass_unique_threshold_list[target]:
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

def main(args):
    logger.info("Loading data.")
    adj_ant_dict = get_mit_states_data(args)
    df = pd.read_csv(args.general_statistics_path)
    df = get_target_df(df, adj_ant_dict, args)
    df_info = pd.read_csv(args.df_info_path)
    embeddings = np.load(args.embeddings_path)

    logger.info("Retrieve seed list.")
    np.random.seed(args.seed)
    seed_list = np.random.randint(10000, size=args.num_runs)
    logger.info(f"seed_list: {seed_list}")
    
    logger.info("="*50)
    silhouette_score_list = []
    all_homogeneity_list = []
    all_completeness_list = []
    all_v_measure_list = []
    
    for run_id in range(args.num_runs):
        seed = seed_list[run_id]
        logger.info(f"Run {run_id}")
        logger.info(f"seed: {seed}")

        logger.info("Filter out bigrams that do not reach the sample_threshold and Sampling.")
        final_df_info = get_clean_df_info(df_info, df, args, seed)
        final_embeddings = embeddings[final_df_info.index]
        final_df_info.to_csv("{}/run{}_df_info.csv".format(args.output_dir, run_id))

        logger.info(f"Clustering with TSNE.")
        tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=seed)
        projections = tsne.fit_transform(final_embeddings)
        logger.info("Output to {}".format(args.output_dir))
        np.save("{}/run{}_projections.npy".format(args.output_dir, run_id), projections)
        
        logger.info("Compute clustering score.")
        target_col, other_col = ("adjective", "noun") if args.cluster_type=="adjective" else ("noun", "adjective")
        unique_target_list = final_df_info[target_col].value_counts()

        target_list, homogeneity_list, completeness_list, v_measure_list  = [],[],[],[]
        for process_id, unique_target in enumerate(unique_target_list.index):
            curr_df = (final_df_info[final_df_info[target_col]==unique_target]).reset_index()
            curr_embeddings = final_embeddings[curr_df.index]
            
            logger.info(f"Progress: {process_id}/{len(unique_target_list)}")
            logger.info(f"Number of target rows: {len(curr_df)}")

            # Retrieve number of neighbors and number of labels
            n_clusters = len(curr_df[other_col].unique())
            labels = curr_df[other_col]

            # Construct kmeans classifier
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
            kmeans_preds = kmeans.fit_predict(curr_embeddings)
            
            homogeneity = homogeneity_score(labels, kmeans_preds)
            completeness = completeness_score(labels, kmeans_preds)
            v_measure = v_measure_score(labels, kmeans_preds)

            target_list.append(unique_target)
            homogeneity_list.append(homogeneity)
            completeness_list.append(completeness)
            v_measure_list.append(v_measure)

        # Output
        res_df = pd.DataFrame(
            {
                "target": target_list, 
                "homogeneity": homogeneity_list,
                "completeness": completeness_list,
                "v_measure": v_measure_list,
            }
        )
        res_df.to_csv("{}/run{}_results.csv".format(args.output_dir, run_id), index=False)
        all_homogeneity_list.append(np.mean(homogeneity_list))
        all_completeness_list.append(np.mean(completeness_list))
        all_v_measure_list.append(np.mean(v_measure_list))
        
        num_unique_clusters = len(final_df_info.adjective.unique()) if args.cluster_type=="ADJ" else len(final_df_info.noun.unique())
        logger.info(f"Number of unique clusters: {num_unique_clusters}")

    logger.info(f"Number of runs: {args.num_runs}")
    logger.info(f"Mean homogeneity scores: {np.mean(all_homogeneity_list)}")
    logger.info(f"Standard deviation homogeneity scores: {np.std(all_homogeneity_list)}")
    logger.info(f"Mean completeness scores: {np.mean(all_completeness_list)}")
    logger.info(f"Standard deviation completeness scores: {np.std(all_completeness_list)}")
    logger.info(f"Mean v_measure scores: {np.mean(all_v_measure_list)}")
    logger.info(f"Standard deviation v_measure scores: {np.std(all_v_measure_list)}")    
    logger.info("Done!")

if __name__ == "__main__":
    args = parseArguments()
    logger = get_logger(args)
    logging_args(args)
    logger.info("Device: {}".format(device))
    
    np.random.seed(args.seed)
    main(args)


