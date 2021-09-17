# Get arguments
while getopts e:c: flag
do
    case "${flag}" in
        e) embedder=${OPTARG};;
        c) cls_type=${OPTARG};;
    esac
done

# Run experiment
python3 -m vlm_lexical_grounding.piqa.piqa_transformer_probing \
    --verbose \
    --num_rows -1 \
    --num_runs 5 \
    --embedder ${embedder} \
    --cls_type ${cls_type} \
    --accumulation_step 32 \
    --learning_rate 1e-4 


