import os
from typing import Dict, OrderedDict
from tqdm import tqdm
import torch
import torch.nn as nn
from typing import Union, Optional, Callable, List, Tuple
from transformers import Trainer
import datasets
import numpy as np
from typing import Optional, List
from datasets import Dataset, DatasetInfo, NamedSplit, DatasetDict
from datasets.table import Table, list_table_cache_files
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader
from transformers import PreTrainedModel, DataCollator, PreTrainedTokenizerBase, EvalPrediction, TrainerCallback
from transformers.trainer_pt_utils import DistributedSamplerWithLoop, get_length_grouped_indices
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler as DistributedLengthGroupedSamplerOri
from transformers.trainer_pt_utils import LengthGroupedSampler as LengthGroupedSamplerOri
# from transformers.trainer_utils import has_length
from transformers.training_args import ParallelMode
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer, 
    SequentialDistributedSampler, 
    nested_concat,
    )
from transformers.utils import logging
from transformers.trainer_utils import denumpify_detensorize, TrainOutput
from training.sampler import sample_by_bald_class_easiness
from training.trainer_base import BaseTrainer


logger = logging.get_logger('Self-training')


class DatasetK(Dataset):
    def __init__(
            self,
            arrow_table: Table,
            info: Optional[DatasetInfo] = None,
            split: Optional[NamedSplit] = None,
            indices_table: Optional[Table] = None,
            fingerprint: Optional[str] = None,
    ):
        self.custom_cache_files = None
        super(DatasetK, self).__init__(arrow_table, info, split, indices_table, fingerprint)


    @property
    def cache_files(self) -> List[dict]:
        """The cache files containing the Apache Arrow table backing the dataset."""
        if self.custom_cache_files:
            return self.custom_cache_files
        cache_files = list_table_cache_files(self._data)
        if self._indices is not None:
            cache_files += list_table_cache_files(self._indices)
        return [{"filename": cache_filename} for cache_filename in cache_files]

    def set_cache_files(self, custom_cache_files):
        self.custom_cache_files = custom_cache_files


# add by wjn
def random_sampling(raw_datasets, num_examples_per_label: Optional[int]=16):
    label_list = raw_datasets["label"] # [0, 1, 0, 0, ...]
    label_dict = dict()
    # 记录每个label对应的样本索引
    for ei, label in enumerate(label_list):
        if label not in label_dict.keys():
            label_dict[label] = list()
        label_dict[label].append(ei)
    # 对于每个类别，随机采样k个样本
    few_example_ids = list()
    for label, eid_list in label_dict.items():
        # examples = deepcopy(eid_list)
        # shuffle(examples)
        idxs = np.random.choice(len(eid_list), size=num_examples_per_label, replace=False)
        selected_eids = [eid_list[i] for i in idxs]
        few_example_ids.extend(selected_eids)
    return few_example_ids

class TeacherTrainer(BaseTrainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            test_key: str = "accuracy"
    ):
        super(TeacherTrainer, self).__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers)
        self.predict_dataset = eval_dataset
        self.test_key = test_key
        # if self.args.do_adv:
        #     self.fgm = FGM(self.model)
        # for callback in callbacks:
        #     callback.trainer = self
        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })
        self.global_step_ = 0
    

    def mc_evaluate(
        self,
        unlabeled_dataset: Optional[Dataset] = None,
        unlabeled_data_num: int = -1,
        description: str = "Evaluate on Unlabeled Data via MC Dropout Uncertainty Estimation",
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        T: int = 30,
        num_classes: int = 0
    ):
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        is_sample = True
        if unlabeled_data_num == -1 or unlabeled_data_num >= len(unlabeled_dataset):
            unlabeled_data_num = len(unlabeled_dataset)
            is_sample = False
        
        if is_sample:
            recalled_examples_idx_list = random_sampling(
                raw_datasets=unlabeled_dataset, 
                num_examples_per_label=unlabeled_data_num // num_classes
            )
            unlabeled_dataset = unlabeled_dataset.select(recalled_examples_idx_list)
            unlabeled_data_num = len(unlabeled_dataset)

        unlabeled_dataloader = self.get_eval_dataloader(unlabeled_dataset)
        model = self._wrap_model(self.model, training=True, dataloader=unlabeled_dataloader) # reset training to True

        batch_size = unlabeled_dataloader.batch_size
        # unlabeled_data_num = self.num_examples(unlabeled_dataloader)
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {unlabeled_data_num}")
        logger.info(f"  Batch size = {batch_size}")

        # world_size = max(1, args.world_size)
        
        # if not prediction_loss_only:
        #     # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
        #     # a batch size to the sampler)
        #     make_multiple_of = None
        #     if hasattr(unlabeled_dataloader, "sampler") and isinstance(unlabeled_dataloader.sampler, SequentialDistributedSampler):
        #         make_multiple_of = unlabeled_dataloader.sampler.batch_size

        model.train() # 开启train模式，允许模型进行Dropout

        if args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = unlabeled_dataloader

        # y_T = np.zeros((T, unlabeled_data_num, num_classes))
        y_T = list()

        for i in tqdm(range(T)):
            y_pred = []

            for step, inputs in enumerate(unlabeled_dataloader):
                _, logits, __ = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
                y_pred.extend(logits.detach().cpu().numpy().tolist())
            # print("y_pred.shape=", torch.Tensor(y_pred).shape) # [n, num_class]
            predict_proba = torch.softmax(torch.Tensor(y_pred).to(logits.device), -1)
            # print("predict_proba.shape=", predict_proba.shape) # [n, num_class]
            # y_T[i] = predict_proba.detach().cpu().numpy().tolist()
            y_T.append(predict_proba.detach().cpu().numpy().tolist())
        
        y_T = np.array(y_T)
        #compute mean
        y_mean = np.mean(y_T, axis=0)
        # print("y_mean.shape=", y_mean.shape) # e.g., (4095, 3) [n, class_num]
        # print("(unlabeled_data_num, num_classes)=", (unlabeled_data_num, num_classes))
        assert y_mean.shape == (unlabeled_data_num, num_classes)

        #compute majority prediction
        y_pred = np.array([np.argmax(np.bincount(row)) for row in np.transpose(np.argmax(y_T, axis=-1))])
        assert y_pred.shape == (unlabeled_data_num,)

        #compute variance
        y_var = np.var(y_T, axis=0)
        assert y_var.shape == (unlabeled_data_num, num_classes)

        return unlabeled_dataset, y_mean, y_var, y_pred, y_T



class RobustTrainer(TeacherTrainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            test_key: str = "accuracy"
    ):
        super(RobustTrainer, self).__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers)
        self.predict_dataset = eval_dataset
        self.test_key = test_key
        # if self.args.do_adv:
        #     self.fgm = FGM(self.model)
        # for callback in callbacks:
        #     callback.trainer = self
        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })
        self.global_step_ = 0
    
    def robust_train(self):
        pass



class SelfTrainer(object):
    def __init__(
        self, 
        base_mode: torch.nn.Module,
        training_args,
        semi_training_args,
        train_dataset: Optional[Dataset]=None,
        unlabeled_dataset: Optional[Dataset]=None,
        eval_dataset=None,
        compute_metrics=None,
        tokenizer=None,
        teacher_data_collator=None,
        student_data_collator=None,
        test_key=None,
        task_type="cls",
        num_classes=0,
    ) -> None:

        logger.info("This is a Self-trainer.")
        
        self.base_model = base_mode
        self.training_args = training_args
        self.semi_training_args = semi_training_args
        self.train_dataset = train_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.tokenizer = tokenizer
        self.teacher_data_collator = teacher_data_collator
        self.student_data_collator = student_data_collator
        self.test_key = test_key
        self.task_type = task_type
        self.num_classes = num_classes

        # self.set_teacher_trainer()
        # self.set_student_trainer()
        self.training_args.per_device_train_batch_size = self.semi_training_args.unlabeled_data_batch_size
        self.teacher_training_epoch = self.semi_training_args.teacher_training_epoch # 最初teacher模型在labeled data上训练的epoch数
        self.teacher_tuning_epoch = self.semi_training_args.teacher_tuning_epoch # 每一轮Self-training时，teacher模型继续在labeled data上tune的epoch数
        self.student_training_epoch = self.semi_training_args.student_training_epoch # 每一轮Self-training时，student模型在pseudo-labeled data上训练的epoch数
        self.self_training_epoch = self.semi_training_args.self_training_epoch # Self-training迭代数
        self.unlabeled_data_num = self.semi_training_args.unlabeled_data_num # self-training每轮迭代时，首先挑选一部分用于计算MC dropout uncertainty。-1表示全部计算uncertainty
        self.pseudo_sample_num_or_ratio = self.semi_training_args.pseudo_sample_num_or_ratio # MC dropout后，从所有计算过uncertainty的unlabeled data上采样的样本比例/数量
        self.student_learning_rate = self.semi_training_args.student_learning_rate
        self.output_dir = self.training_args.output_dir

    def get_teacher_trainer(
        self, 
        base_model: torch.nn.Module, 
        num_train_epochs: int,
        output_dir: str = None,
        ):
        training_args = self.training_args
        training_args.num_train_epochs = num_train_epochs
        if output_dir is not None:
            training_args.output_dir = output_dir
        # 初始化Teacher训练器
        teacher_trainer = TeacherTrainer(
            model=base_model,
            args=training_args,
            train_dataset=self.train_dataset if self.training_args.do_train else None,
            eval_dataset=self.eval_dataset if self.training_args.do_eval else None,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
            data_collator=self.teacher_data_collator,
            test_key=self.test_key,
        )
        return teacher_trainer

    
    def get_student_trainer(
        self, 
        base_model: torch.nn.Module, 
        num_train_epochs: int, 
        student_learning_rate: float,
        pseudo_labeled_dataset: Optional[Dataset] = None, 
        output_dir: str = None,
        ):
        training_args = self.training_args
        training_args.num_train_epochs = num_train_epochs
        training_args.learning_rate = student_learning_rate
        if output_dir is not None:
            training_args.output_dir = output_dir
        # 初始化Student训练器
        student_trainer = RobustTrainer(
            model=base_model,
            args=training_args,
            train_dataset=pseudo_labeled_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
            data_collator=self.student_data_collator,
            test_key=self.test_key,
        )
        return student_trainer

    def freeze_backbone(self, model: torch.nn.Module, use_pe: bool=False):
        try:
            model.freeze_backbone(use_pe=use_pe)
        except:
            pass
        return model


    def train(self, resume_from_checkpoint=None):
        # 首先对Teacher模型在labeled data上进行full parameter fine-tuning
        logger.info("*"*66)
        logger.info("* Training teacher model over labeled data before self-training. *")
        logger.info("*"*66)
        print("*"*66)
        print("* Training teacher model over labeled data before self-training. *")
        print("*"*66)

        teacher_model = self.base_model
        teacher_model = self.freeze_backbone(teacher_model, use_pe=False)
        teacher_trainer: TeacherTrainer = self.get_teacher_trainer(base_model=teacher_model, num_train_epochs=self.teacher_training_epoch)
        if resume_from_checkpoint is not None:
            teacher_trainer._load_from_checkpoint(resume_from_checkpoint)
        teacher_trainer.train()
        teacher_model.load_state_dict(torch.load(os.path.join(teacher_trainer.state.best_model_checkpoint, "pytorch_model.bin")))
        teacher_trainer.model = teacher_model

        # 原始的训练结果
        metrics = teacher_trainer.evaluate()
        convention_result = metrics["eval_{}".format(self.test_key)]

        logger.info("*"*50)
        logger.info("* Conventional fine-tuning metric: {}. *".format(convention_result))
        logger.info("*"*50)
        print("*"*50)
        print("* Conventional fine-tuning metric: {}. *".format(convention_result))
        print("*"*50)

        logger.info("*"*30)
        logger.info("* Starting Self-training ... *")
        logger.info("*"*30)
        print("*"*30)
        print("* Starting Self-training ... *")
        print("*"*30)

        best_test_metric = None
        best_self_training_iteration = None
        best_teacher_model = None

        # 多轮Teacher-Student迭代训练
        for iter in range(self.self_training_epoch):

            logger.info("*"*34)
            logger.info("* Self-training {}-th iteration *".format(iter))
            logger.info("*"*34)
            print("*"*34)
            print("* Self-training {}-th iteration *".format(iter))
            print("*"*34)

            # if iter == self.self_training_epoch - 1:
            #     break


            # # Teacher模型在labeled data上进行parameter-efficient tuning
            # if iter > 0:
            #     logger.info("*"*80)
            #     logger.info("* Tuning the teacher model on labeled data at {}-th self-training iteration. *".format(iter))
            #     logger.info("*"*80)
            #     print("*"*80)
            #     print("* Tuning the teacher model on labeled data at {}-th self-training iteration. *".format(iter))
            #     print("*"*80)

            #     teacher_model = self.freeze_backbone(teacher_model, use_pe=True)
            #     # teacher_trainer: TeacherTrainer = self.get_teacher_trainer(base_model=teacher_model, num_train_epochs=self.teacher_tuning_epoch)
            #     teacher_trainer.train()
            #     teacher_model.load_state_dict(torch.load(os.path.join(teacher_trainer.state.best_model_checkpoint, "pytorch_model.bin")))
            #     teacher_trainer.model = teacher_model
            
            # Teacher模型在unlabeled data上获取pseudo-labeled data，并根据uncertainty estimation进行采样
            logger.info("*"*72)
            logger.info("Obtaining pseudo-labeled data and uncertainty estimation via MC dropout.")
            logger.info("*"*72)
            print("*"*72)
            print("Obtaining pseudo-labeled data and uncertainty estimation via MC dropout.")
            print("*"*72)

            unlabeled_dataset, y_mean, y_var, y_pred, y_T = teacher_trainer.mc_evaluate(
                unlabeled_dataset=self.unlabeled_dataset, 
                unlabeled_data_num=self.unlabeled_data_num,
                T=30, 
                num_classes=self.num_classes
                )
            
            logger.info("*"*42)
            logger.info("* Sampling reliable pseudo-labeled data. *")
            logger.info("*"*42)
            print("*"*42)
            print("* Sampling reliable pseudo-labeled data. *")
            print("*"*42)
            
            X_batch, y_batch, _ = sample_by_bald_class_easiness(
                tokenizer=self.tokenizer, 
                X=unlabeled_dataset, 
                y_mean=y_mean, 
                y_var=y_var, 
                y=y_pred, 
                num_samples=int(y_pred.shape[0] * self.pseudo_sample_num_or_ratio) if self.pseudo_sample_num_or_ratio <= 1.0 else int(self.pseudo_sample_num_or_ratio), 
                num_classes=self.num_classes, 
                y_T=y_T)
            pseudo_labeled_examples = X_batch
            pseudo_labeled_examples["label"] = y_batch
            
            # 生成pseudo-labeled dataset，并与labeled data混合
            # pseudo_labeled_dataset = DatasetDict()
            pseudo_labeled_dataset = DatasetK.from_dict(pseudo_labeled_examples)
            for i in range(len(self.train_dataset)):
                pseudo_labeled_dataset = pseudo_labeled_dataset.add_item(self.train_dataset[i])

            # 初始化一个新的Student模型，并让Student模型在pseudo-labeled data上进行鲁棒学习
            logger.info("*"*56)
            logger.info("* Training a new student model on pseudo-labeled data. *")
            logger.info("*"*56)
            print("*"*56)
            print("* Training a new student model on pseudo-labeled data. *")
            print("*"*56)
            
            student_model = self.base_model
            student_model = self.freeze_backbone(student_model, use_pe=True)
            student_trainer: RobustTrainer = self.get_student_trainer(
                base_model=self.base_model, 
                num_train_epochs=self.student_training_epoch,
                student_learning_rate=self.student_learning_rate,
                pseudo_labeled_dataset=pseudo_labeled_dataset,
                output_dir=os.path.join(self.output_dir, "student_iter_{}".format(iter))
            )
            student_trainer.train()
            student_model.load_state_dict(torch.load(os.path.join(student_trainer.state.best_model_checkpoint, "pytorch_model.bin")))

            # 将Student模型参数赋给Teacher，作为下一轮训练的Teacher初始化
            logger.info("*"*64)
            logger.info("* Initializing a new teacher model from trained student model. *")
            logger.info("*"*64)
            print("*"*64)
            print("* Initializing a new teacher model from trained student model. *")
            print("*"*64)
            teacher_model = student_model
            teacher_trainer = student_trainer
            # teacher_trainer: TeacherTrainer = self.get_teacher_trainer(
            #     base_model=student_model, 
            #     num_train_epochs=self.teacher_tuning_epoch, 
            #     output_dir=os.path.join(self.output_dir, "teacher_iter_{}".format(iter))
            # )


            # 获得Teacher模型在测试集上的效果
            # teacher_trainer.model = teacher_model
            metrics = teacher_trainer.evaluate()
            # print("metrics=", metrics)
            '''
            e.g., {'eval_loss': 0.6926815509796143, 'eval_accuracy': 0.5234657039711191, 'eval_runtime': 0.7267, 'eval_samples_per_second': 381.161, 'eval_steps_per_second': 48.161, 'epoch': 1.0}
            '''
            logger.info("*"*60)
            logger.info("* The testing result of teacher model is {} result: {} *".format(self.test_key, metrics["eval_{}".format(self.test_key)]))
            logger.info("*"*60)
            print("*"*60)
            print("* The testing result of teacher model is {} result: {} *".format(self.test_key, metrics["eval_{}".format(self.test_key)]))
            print("*"*60)

            if best_test_metric is None or best_test_metric < metrics["eval_{}".format(self.test_key)]:
                best_test_metric = metrics["eval_{}".format(self.test_key)]
                best_self_training_iteration = iter
                best_teacher_model = teacher_model
                logger.info("The best teacher model at {}-th self-training iteration.".format(best_self_training_iteration))
                logger.info("The best teacher model testing result is {}.".format(best_test_metric))
                print("The best teacher model at {}-th self-training iteration.".format(best_self_training_iteration))
                print("The best teacher model testing result is {}.".format(best_test_metric))
            
        
        logger.info("********** Finishing Self-training **********")
        logger.info("The best teacher model at {}-th self-training iteration.".format(best_self_training_iteration))
        logger.info("The best teacher model testing result is {}.".format(best_test_metric))
        print("********** Finishing Self-training **********")
        print("The best teacher model at {}-th self-training iteration.".format(best_self_training_iteration))
        print("The best teacher model testing result is {}.".format(best_test_metric))

        
        # 根据当前最好的Teacher模型，在全部的unlabeled data上打伪标签，并进行mc dropout（样本数量最多不超过50000）
        if self.semi_training_args.post_student_train:
            
            logger.info("********** Post training **********")
            print("********** Post training **********")

            teacher_trainer: TeacherTrainer = self.get_teacher_trainer(
                base_model=best_teacher_model, 
                num_train_epochs=self.teacher_tuning_epoch, 
                output_dir=os.path.join(self.output_dir, "teacher_iter_post")
            )

            unlabeled_dataset, y_mean, y_var, y_pred, y_T = teacher_trainer.mc_evaluate(
                unlabeled_dataset=self.unlabeled_dataset, 
                unlabeled_data_num=20480,
                T=5, 
                num_classes=self.num_classes
                )
            
            post_sample_num = int(y_pred.shape[0] * 0.5)
            
            X_batch, y_batch, _ = sample_by_bald_class_easiness(
                tokenizer=self.tokenizer, 
                X=unlabeled_dataset, 
                y_mean=y_mean, 
                y_var=y_var, 
                y=y_pred, 
                num_samples=post_sample_num, 
                num_classes=self.num_classes, 
                y_T=y_T)
            pseudo_labeled_examples = X_batch
            pseudo_labeled_examples["label"] = y_batch
            # 生成pseudo-labeled dataset
            # pseudo_labeled_dataset = DatasetDict()
            pseudo_labeled_dataset = DatasetK.from_dict(pseudo_labeled_examples)


            # 初始化一个新的Student模型，并让Student模型在pseudo-labeled data上进行鲁棒学习
            logger.info("*"*56)
            logger.info("* Training a new student model on pseudo-labeled data. *")
            logger.info("*"*56)
            print("*"*56)
            print("* Training a new student model on pseudo-labeled data. *")
            print("*"*56)
            
            student_model = self.base_model
            student_model = self.freeze_backbone(student_model, use_pe=True)
            student_trainer: RobustTrainer = self.get_student_trainer(
                base_model=self.base_model, 
                num_train_epochs=self.student_training_epoch if len(pseudo_labeled_dataset) <= 4096 else int(self.student_training_epoch / 2),
                student_learning_rate=self.student_learning_rate,
                pseudo_labeled_dataset=pseudo_labeled_dataset,
                output_dir=os.path.join(self.output_dir, "student_iter_{}".format(iter))
            )
            student_trainer.train()
            student_model.load_state_dict(torch.load(os.path.join(student_trainer.state.best_model_checkpoint, "pytorch_model.bin")))

            metrics = student_trainer.evaluate()
            post_metric = metrics["eval_{}".format(self.test_key)]


        print("*"*68)
        print("Finishing all the processes, the results are shown in the following:")
        print("Conventional fine-tuning {} metric: {}".format(self.test_key, convention_result))
        print("Best self-training {} metric: {}".format(self.test_key, best_test_metric))
        if self.semi_training_args.post_student_train:
            print("Post training {} metric: {}".format(self.test_key, post_metric))
        print("*"*68)

        return TrainOutput(teacher_trainer.state.global_step, 0.0, metrics)