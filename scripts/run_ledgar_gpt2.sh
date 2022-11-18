#CUDA_VISIBLE_DEVICES=0 python -m experiments.ledgar --model_name_or_path gpt2-xl --do_lower_case False  --output_dir logs/ledgar/gpt2-xl/seed_1 --do_train --do_eval --do_pred --overwrite_output_dir --load_best_model_at_end --metric_for_best_model micro-f1 --greater_is_better True --evaluation_strategy epoch --save_strategy epoch --save_total_limit 5 --num_train_epochs 1 --learning_rate 6e-6 --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --seed 1 --fp16 --fp16_full_eval  --max_seq_length 512 --gradient_checkpoint True

CUDA_VISIBLE_DEVICES=0 python -m experiments.ledgar \
    --model_name_or_path gpt2-xl \
    --do_lower_case False  
    --output_dir logs/ledgar/gpt2-xl/seed_1 \
    --do_train \
    --do_eval \
    --do_pred \
    --overwrite_output_dir \
    --load_best_model_at_end \
    --metric_for_best_model micro-f1 \
    --greater_is_better True \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 5 \
    --num_train_epochs 1 \
    --learning_rate 6e-6 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --seed 1 \
    --fp16 \
    --fp16_full_eval  \
    --max_seq_length 512 \
    --gradient_checkpoint True\
    --max_train_samples 1000 \
    --pad_to_max_length False