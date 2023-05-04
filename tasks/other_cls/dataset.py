import torch
from torch.utils import data
from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import logging
from typing import Optional


# add by wjn
def random_sampling(raw_datasets: load_dataset, data_type: str="train", num_examples_per_label: Optional[int]=16, label_set: list=None):
    assert data_type in ["train", "dev", "test"]

    label_list = raw_datasets[data_type]["label"] if "label" in raw_datasets[data_type].features else raw_datasets[data_type]["coarse_label"] # [0, 1, 0, 0, ...]
    label_dict = dict()
    # 记录每个label对应的样本索引
    for ei, label in enumerate(label_list):
        # print(label)
        # if int(label) not in label_set: # 部分task存在-1的label，需要剔除
        #     continue
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

    # 保存没有被选中的example id
    num_examples = len(label_list)
    un_selected_examples_ids = [idx for idx in range(num_examples) if idx not in few_example_ids]
    return few_example_ids, un_selected_examples_ids

task_to_test_key = {
    "movie_rationales": "accuracy",
    "cr": "accuracy",
    "snli": "accuracy",
    "ag_news": "accuracy",
    "yelp_polarity": "accuracy",
    "trec": "accuracy",
}

task_to_keys = {
    "movie_rationales": ("review", None),
    "cr": ("sentence", None),
    "snli": ("premise", "hypothesis"),
    "ag_news": ("text", None),
    "yelp_polarity": ("text", None),
    "trec": ("text", None),
}

task_to_template = {
    "movie_rationales": [{"prefix_template": "", "suffix_template": "It was <mask> !"}, None],
    "cr": [{"prefix_template": "", "suffix_template": "It was <mask> ."}, None],
    "snli": [None, {"prefix_template": " ? <mask> , ", "suffix_template": ""}],
    "ag_news": [{"prefix_template": "<mask> News: ", "suffix_template": ""}, None],
    "yelp_polarity": [{"prefix_template": "", "suffix_template": "It was <mask> ."}, None],
    "trec": [{"prefix_template": "<mask> question: ", "suffix_template": ""}, None],
    
}

# add by wjn
label_words_mapping = {
    "movie_rationales": {"NEG": ["terrible"], "POS": ["great"]},
    "cr": {"negative": ["terrible"], "positive": ["great"]},
    "snli": {"contradiction": ["No"], "neutral": ["Maybe"], "entailment": ["Yes"]},
    "ag_news": {"World" : ["World"], "Sports": ["Sports"], "Business": ["Business"], "Sci/Tech": ["Tech"]},
    "yelp_polarity": {"1": ["bad"], "2": ["great"]},
    "trec": {"DESC": ["Description"], "ENTY": ["Entity"], "ABBR": ["Abbreviation"], "HUM": ["Human"], "NUM": ["Numeric"], "LOC": ["Location"]}
}


logger = logging.getLogger(__name__)


class OtherCLSDataset():
    def __init__(
        self, 
        tokenizer: AutoTokenizer, 
        data_args, 
        training_args, 
        semi_training_args=None,
        use_prompt=None
    ) -> None:
        super().__init__()
        raw_datasets = load_dataset(data_args.dataset_name)
        self.tokenizer = tokenizer
        self.data_args = data_args
        #labels
        self.is_regression = False
        if not self.is_regression:
            if self.data_args.dataset_name == "trec":
                self.label_list = raw_datasets["train"].features["coarse_label"].names
            else:
                self.label_list = raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
        else:
            self.num_labels = 1
        # print("self.label_list=", self.label_list)

        # === generate template ===== add by wjn
        self.use_prompt = False
        if data_args.dataset_name in task_to_template.keys():
            self.use_prompt = use_prompt
        
        if self.use_prompt:
            if 't5' in type(tokenizer).__name__.lower():
                self.special_token_mapping = {
                    'cls': 3, 'mask': 32099, 'sep': tokenizer.eos_token_id,
                    'sep+': tokenizer.eos_token_id,
                    'pseudo_token': tokenizer.unk_token_id
                }
            else:
                self.special_token_mapping = {
                    'cls': tokenizer.cls_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id,
                    'sep+': tokenizer.sep_token_id,
                    'pseudo_token': tokenizer.unk_token_id
                }
            self.template = task_to_template[data_args.dataset_name] # dict

        # === generate label word mapping ===== add by wjn
        if self.use_prompt:
            assert data_args.dataset_name in label_words_mapping.keys(), "You must define label word mapping for the task {}".format(data_args.dataset_name)
            self.label_to_word = label_words_mapping[data_args.dataset_name] # e.g., {"0": ["great"], "1": [bad]}
            self.label_to_word = {label: label_word[0] if type(label_word) == list else label_word for label, label_word in self.label_to_word.items()}

            for key in self.label_to_word:
                # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
                if self.label_to_word[key][0] not in ['<', '[', '.', ',']:
                    # Make sure space+word is in the vocabulary
                    # assert len(tokenizer.tokenize(' ' + self.label_to_word[key])) == 1
                    self.label_to_word[key] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + self.label_to_word[key])[0])
                else:
                    self.label_to_word[key] = tokenizer.convert_tokens_to_ids(self.label_to_word[key])
                logger.info("Label {} to word {} ({})".format(key, tokenizer._convert_id_to_token(self.label_to_word[key]), self.label_to_word[key]))

            if len(self.label_list) > 1:
                self.label_word_list = [self.label_to_word[label] for label in self.label_list]
            else:
                # Regression task
                # '0' represents low polarity and '1' represents high polarity.
                self.label_word_list = [self.label_to_word[label] for label in ['0', '1']]
        # =============

        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = task_to_keys[data_args.dataset_name]

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        if not self.is_regression:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        raw_datasets = raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        if training_args.do_train:
            self.train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval and data_args.dataset_name not in ["ag_news", "yelp_polarity", "trec"]:
            self.eval_dataset = raw_datasets["validation_matched" if data_args.dataset_name == "mnli" else "validation"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            self.predict_dataset = raw_datasets["test_matched" if data_args.dataset_name == "mnli" else "test"]
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))
        
        if data_args.dataset_name in ["ag_news", "yelp_polarity", "trec"]:
            self.eval_dataset = self.predict_dataset
         # add by wjn 
        self.unlabeled_dataset = None
        if semi_training_args.use_semi is True:
            assert data_args.num_examples_per_label is not None and data_args.num_examples_per_label != -1
        
        # 随机采样few-shot training / dev data（传入label_list，对每个label进行采样，最后得到索引列表）
        if data_args.num_examples_per_label is not None and data_args.num_examples_per_label != -1:
            train_examples_idx_list, un_selected_idx_list = random_sampling(
                raw_datasets=raw_datasets, 
                data_type="train", 
                num_examples_per_label=data_args.num_examples_per_label,
                label_set=list(self.id2label.keys())
            )
            self.all_train_dataset = self.train_dataset
            self.train_dataset = self.all_train_dataset.select(train_examples_idx_list)
            print("Randomly sampling {}-shot training examples for each label. Total examples number is {}".format(
                data_args.num_examples_per_label, 
                len(self.train_dataset)
                ))
            
            if semi_training_args.use_semi is True:
                self.unlabeled_dataset = self.all_train_dataset.select(un_selected_idx_list)
                print("The number of unlabeled data is {}".format(len(self.unlabeled_dataset)))

        self.metric = load_metric("./metrics/accuracy", data_args.dataset_name)

        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        
        # ==== define test key ====== add by wjn
        self.test_key = task_to_test_key[data_args.dataset_name]


    def preprocess_function(self, examples):
        # add by wjn
        if self.data_args.dataset_name == "trec":
            examples["label"] = []
            for label in examples["coarse_label"]:
                examples["label"].append(label)
                

        # add by wjn
        # adding prompt into each example
        if self.use_prompt:
            # if use prompt, insert template into example
            examples = self.prompt_preprocess_function(examples)
        
        # Tokenize the texts
        args = (
            (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
        )
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

        if self.use_prompt:
            mask_pos = []
            for input_ids in result["input_ids"]:
                mask_pos.append(input_ids.index(self.special_token_mapping["mask"]))
            result["mask_pos"] = mask_pos
        
        return result
    
    # add by wjn
    # process data for prompt (add template)
    def prompt_preprocess_function(self, examples):
        
        def replace_mask_token(template):
            return template.replace("<mask>", self.tokenizer.convert_ids_to_tokens(self.special_token_mapping["mask"]))
        
        sequence1_prefix_template = replace_mask_token(self.template[0]["prefix_template"] if self.template[0] is not None else "")
        sequence1_suffix_template = replace_mask_token(self.template[0]["suffix_template"] if self.template[0] is not None else "")
        sequence2_prefix_template = replace_mask_token(self.template[1]["prefix_template"] if self.template[1] is not None else "")
        sequence2_suffix_template = replace_mask_token(self.template[1]["suffix_template"] if self.template[1] is not None else "")
        example_num = len(examples[self.sentence1_key])
        for example_id in range(example_num):
            sequence1 = examples[self.sentence1_key][example_id]
            if self.sentence2_key is None:
                sequence1 = sequence1[:self.data_args.max_seq_length - len(sequence1_suffix_template) - 10]
            examples[self.sentence1_key][example_id] = "{}{}{}".format(
                sequence1_prefix_template, sequence1, sequence1_suffix_template)
            
            if self.sentence2_key is not None:
                sequence2 = examples[self.sentence2_key][example_id]
                sequence2 = sequence2[:self.data_args.max_seq_length - len(sequence1) - len(sequence1_prefix_template) - len(sequence1_suffix_template) - len(sequence2_prefix_template)- 10]
                examples[self.sentence2_key][example_id] = "{}{}{}".format(
                    sequence2_prefix_template, sequence2, sequence2_suffix_template)
        return examples

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
        if self.data_args.dataset_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

