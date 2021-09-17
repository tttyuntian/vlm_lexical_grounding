EMBEDDER=$1

# Run experiment
python3 -m vlm_lexical_grounding.piqa.precompute_sentence_embedding \
    --verbose \
    --num_rows -1 \
    --embedder $EMBEDDER \
