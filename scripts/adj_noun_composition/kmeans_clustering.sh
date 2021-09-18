# Get arguments
while getopts e: flag
do
    case "${flag}" in
        e) embedder=${OPTARG};;
    esac
done

# Run experiment
python3 -m vlm_lexical_grounding.adj_noun_composition.kmean_clustering \
    --verbose \
    --num_runs 5 \
    --seed 1123 \
    --cluster_type "NOUN" \
    --embedder ${embedder} \
    --bigram_occur_threshold 10 \
    --unique_bigram_threshold 10 \
    --sample_threshold 20 \
    --method "kmeans"

