export TASK_NAME=other_cls
export DATASET_NAME=movie_rationales
export CUDA_VISIBLE_DEVICES=0

bs=4
lr=5e-5
dropout=0.1
psl=16
epoch=100

python3 run.py \
  --model_name_or_path /wjn/pre-trained-lm/roberta-large \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --overwrite_cache \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 256 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-roberta/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 42 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --num_examples_per_label 16 \
  --prompt_ptuning \
  # --use_pe
  # --adapter \
  # --adapter_dim 256 \
  # --adapter_choice list \

