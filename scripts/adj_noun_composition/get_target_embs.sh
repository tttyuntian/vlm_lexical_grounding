# Get arguments
while getopts e: flag
do
    case "${flag}" in
        e) embedder=${OPTARG};;
    esac
done

# Run experiment
python3 -m vlm_lexical_grounding.adj_noun_composition.get_target_embs \
    --verbose \
    --embedder ${embedder} \
    --bigram_occur_threshold 10 \
    --unique_bigram_threshold 10 \

