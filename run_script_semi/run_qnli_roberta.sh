export TASK_NAME=glue
export DATASET_NAME=qnli
export CUDA_VISIBLE_DEVICES=4
# export CUDA_VISIBLE_DEVICES=2
# export CUDA_VISIBLE_DEVICES=5

bs=4
lr=1e-5
student_lr=5e-3
dropout=0.1
psl=16
student_psl=15
tea_train_epoch=100
tea_tune_epoch=20
stu_train_epoch=100
self_train_epoch=15

pe_type=head_only

rm -rf checkpoints/self-training/$DATASET_NAME-roberta/$pe_type/iteration

python3 run.py \
  --model_name_or_path /wjn/pre-trained-lm/roberta-large \
  --resume_from_checkpoint checkpoints/self-training/$DATASET_NAME-roberta/$pe_type/checkpoint-60 \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --overwrite_cache \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $tea_train_epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/self-training/$DATASET_NAME-roberta/$pe_type \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 42 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --save_total_limit=4 \
  --load_best_model_at_end \
  --metric_for_best_model accuracy \
  --report_to none \
  --num_examples_per_label 16 \
  --$pe_type \
  --use_semi \
  --unlabeled_data_num 4096 \
  --unlabeled_data_batch_size 16 \
  --teacher_training_epoch $tea_train_epoch \
  --teacher_tuning_epoch $tea_tune_epoch \
  --student_training_epoch $stu_train_epoch \
  --self_training_epoch $self_train_epoch \
  --pseudo_sample_num_or_ratio 1024 \
  --student_pre_seq_len $student_psl \
  --post_student_train
 