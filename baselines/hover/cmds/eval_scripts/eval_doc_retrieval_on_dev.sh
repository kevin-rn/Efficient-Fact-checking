claim_name=$1

if [ "$2" = "modified" ]; then
    python3 run_hover.py \
    --dataset_name $claim_name \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --sub_task doc_retrieval \
    --do_eval \
    --do_lower_case \
    --output_dir exp1.0 \
    --max_seq_length 200  \
    --max_query_length 60  \
    --ckpt_to_evaluate 900 \
    --modified
else
    python3 run_hover.py \
    --dataset_name $claim_name \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --sub_task doc_retrieval \
    --do_eval \
    --do_lower_case \
    --output_dir exp1.0 \
    --max_seq_length 200  \
    --max_query_length 60  \
    --ckpt_to_evaluate 900
fi