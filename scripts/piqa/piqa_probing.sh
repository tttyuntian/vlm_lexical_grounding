# Get arguments
while getopts e:c: flag
do
    case "${flag}" in
        e) embedder=${OPTARG};;
        c) cls_type=${OPTARG};;
    esac
done

# Run experiment
python3 -m vlm_lexical_grounding.piqa.piqa_probing \
    --verbose \
    --num_rows 64 \
    --num_runs 5 \
    --embedder ${embedder} \
    --cls_type ${cls_type}

