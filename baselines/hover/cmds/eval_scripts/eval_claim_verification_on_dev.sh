claim_name=$1
latest="$(ls -td out/$claim_name/exp1.0/claim_verification/checkpoint-*/ | head -1)"
checkpoint_number=$(basename "$latest" | sed 's/checkpoint-//')

python3 run_hover.py \
--dataset_name $claim_name \
--model_type bert \
--model_name_or_path bert-base-uncased \
--sub_task claim_verification \
--do_eval \
--do_lower_case \
--output_dir exp1.0 \
--max_seq_length 200  \
--max_query_length 60  \
--ckpt_to_evaluate $checkpoint_number 

python3 evaluate_metrics.py \
--dataset_name $claim_name
