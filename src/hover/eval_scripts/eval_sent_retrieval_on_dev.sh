claim_name=$1
latest="$(ls -td out/$claim_name/exp1.0/sent_retrieval/checkpoint-*/ | head -1)"
checkpoint_number=$(basename "$latest" | sed 's/checkpoint-//')

python3 -m src.hover.run_hover \
    --dataset_name $claim_name \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --sub_task sent_retrieval \
    --do_eval \
    --do_lower_case \
    --output_dir exp1.0 \
    --max_seq_length 200  \
    --max_query_length 60  \
    --ckpt_to_evaluate $checkpoint_number \
    --predict_file hover_dev_sent_retrieval.json