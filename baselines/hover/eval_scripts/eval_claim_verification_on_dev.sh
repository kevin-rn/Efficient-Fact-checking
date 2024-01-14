python3 run_hover.py \
--dataset_name hover \
--model_type bert \
--model_name_or_path bert-base-uncased \
--sub_task claim_verification \
--do_eval \
--do_lower_case \
--output_dir exp1.0 \
--max_seq_length 200  \
--max_query_length 60  \
--ckpt_to_evaluate 2000 

python3 evaluate_metrics.py