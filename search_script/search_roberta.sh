export TASK_NAME=other_cls # glue / superglue / other_cls
export DATASET_NAME=yelp_polarity
export CUDA_VISIBLE_DEVICES=3

bs=4
dropout=0.1
shot=16 # shot=-1 means full data
seq_len=128

paradigm=prompt_adapter # parameter-efficient paradigm
is_pe=use_pe 
is_pe=no_pe

# prompt_only / head_only w. use_pe (lr: 5e-4 5e-3 7e-3 1e-2)
# prompt_only / head_only w/o. use_pe (lr: 5e-6 1e-5 5e-4 5e-3)
# prompt_prefix / head_prefix w. use_pe (lr: 5e-4 5e-3 7e-3 1e-2)
# prompt_prefix / head_prefix w/o. use_pe (lr: 5e-6 1e-5 5e-4 5e-3)
# prompt_ptuning / head_ptuning w. use_pe (lr: 5e-4 5e-3 7e-3 1e-2)
# prompt_ptuning / head_ptuning w/o. use_pe (lr: 5e-6 1e-5 5e-4 5e-3)
# prompt_adapter / head_adapter w. use_pe (lr: 3e-6 5e-6 1e-5 5e-5)
# prompt_adapter / head_adapter w/o. use_pe (lr: 3e-6 5e-6 1e-5 5e-5)

# if choose use_pe: lr can be set among 5e-4 5e-3 7e-3 1e-2
# if not choose use_ps, lr can be set among 5e-6 1e-5 5e-5 5e-4

if [ "$paradigm" = "head_only" ] || [ "$paradigm" = "prompt_only" ]; then
  if [ "$is_pe" = "use_pe" ]; then
    lr_list=(5e-4 5e-3 7e-3 1e-2)
  elif [ "$is_pe" = "no_pe" ]; then
    lr_list=(5e-6 1e-5 5e-4 5e-3)
  fi

elif [ "$paradigm" = "head_prefix" ] || [ "$paradigm" = "prompt_prefix" ]; then
  if [ "$is_pe" = "use_pe" ]; then
    lr_list=(5e-4 5e-3 7e-3 1e-2)
  elif [ "$is_pe" = "no_pe" ]; then
    lr_list=(5e-6 1e-5 5e-4 5e-3)
  fi

elif [ "$paradigm" = "head_ptuning" ] || [ "$paradigm" = "prompt_ptuning" ]; then
  if [ "$is_pe" = "use_pe" ]; then
    lr_list=(5e-4 5e-3 7e-3 1e-2)
  elif [ "$is_pe" = "no_pe" ]; then
    lr_list=(5e-6 1e-5 5e-4 5e-3)
  fi

elif [ "$paradigm" = "head_adapter" ] || [ "$paradigm" = "prompt_adapter" ]; then
  if [ "$is_pe" = "use_pe" ]; then
    lr_list=(3e-6 5e-6 1e-5 5e-5)
  elif [ "$is_pe" = "no_pe" ]; then
    lr_list=(3e-6 5e-6 1e-5 5e-5)
  fi

fi

if [ "$DATASET_NAME" = "boolq" ]; then
  seq_len=384
elif [ "$DATASET_NAME" = "movie_rationales" ]; then
  seq_len=256
fi

for lr in "${lr_list[@]}"
do
  for psl in 4 8 16 32 64 128
  do 
    for epoch in 20 60 100
    do
      python3 run.py \
        --model_name_or_path /wjn/pre-trained-lm/roberta-large \
        --task_name $TASK_NAME \
        --dataset_name $DATASET_NAME \
        --overwrite_cache \
        --do_train \
        --do_eval \
        --max_seq_length $seq_len \
        --per_device_train_batch_size $bs \
        --learning_rate $lr \
        --num_train_epochs $epoch \
        --pre_seq_len $psl \
        --output_dir checkpoints/$DATASET_NAME-roberta-search/$paradigm/$is_pe/$DATASET_NAME-$epoch-$lr-$psl/ \
        --overwrite_output_dir \
        --hidden_dropout_prob $dropout \
        --seed 42 \
        --save_strategy no \
        --evaluation_strategy epoch \
        --num_examples_per_label $shot \
        --prompt_adapter \
        # --use_pe
    done
  done
done
# python3 search.py $DATASET_NAME roberta $paradigm $is_pe