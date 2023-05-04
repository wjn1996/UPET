import logging
import os
import random
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from model.utils import get_model, TaskType
from tasks.superglue.dataset import SuperGlueDataset
from training.trainer_base import BaseTrainer
from training.trainer_exp import ExponentialTrainer
from training.self_trainer import SelfTrainer

logger = logging.getLogger(__name__)

def get_trainer(args):
    model_args, data_args, training_args, semi_training_args, _ = args

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    # add by wjn check if use prompt template
    use_prompt = False
    if model_args.prompt_prefix or model_args.prompt_ptuning or model_args.prompt_adapter or model_args.prompt_only:
        use_prompt = True

    dataset = SuperGlueDataset(tokenizer, data_args, training_args, semi_training_args=semi_training_args, use_prompt=use_prompt)

    data_args.label_word_list = None # add by wjn
    if use_prompt:
        data_args.label_word_list = dataset.label_word_list # add by wjn

    if training_args.do_train:
        for index in random.sample(range(len(dataset.train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {dataset.train_dataset[index]}.")

    if not dataset.multiple_choice:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )

    if not dataset.multiple_choice:
        model = get_model(data_args, model_args, TaskType.SEQUENCE_CLASSIFICATION, config)
    else:
        model = get_model(data_args, model_args, TaskType.MULTIPLE_CHOICE, config)


    # Initialize our Trainer

    if semi_training_args.use_semi:
        model_args.pre_seq_len = semi_training_args.student_pre_seq_len
        student_model = get_model(data_args, model_args, TaskType.SEQUENCE_CLASSIFICATION, config)
        trainer = SelfTrainer(
            teacher_base_model=model,
            student_base_model=student_model,
            training_args=training_args,
            semi_training_args=semi_training_args,
            train_dataset=dataset.train_dataset if training_args.do_train else None,
            unlabeled_dataset=dataset.unlabeled_dataset,
            eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
            compute_metrics=dataset.compute_metrics,
            tokenizer=tokenizer,
            teacher_data_collator=dataset.data_collator,
            student_data_collator=dataset.data_collator,
            test_key=dataset.test_key,
            task_type="cls",
            num_classes=len(dataset.label2id),
        )

        return trainer, None

    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        test_key=dataset.test_key
    )

    return trainer, None
