# Get arguments
while getopts e: flag
do
    case "${flag}" in
        e) embedder=${OPTARG};;
    esac
done

# Run experiment
python3 -m vlm_lexical_grounding.adj_noun_composition.adjective_probing \
    --verbose \
    --embedder ${embedder} \
    --cluster_type "NOUN" \

