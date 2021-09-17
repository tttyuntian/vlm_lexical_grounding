# Get arguments
while getopts e: flag
do
    case "${flag}" in
        e) embedder=${OPTARG};;
    esac
done

# Run experiment
python3 -m vlm_lexical_grounding.piqa.precompute_sentence_embedding \
    --verbose \
    --num_rows -1 \
    --embedder $embedder \
