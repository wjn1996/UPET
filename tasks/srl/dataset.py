from typing import OrderedDict
import torch
from torch.utils import data
from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoConfig
import numpy as np
from typing import Optional

# add by wjn
def random_sampling(raw_datasets: load_dataset, data_type: str="train", num_examples_per_label: Optional[int]=16):
    assert data_type in ["train", "dev", "test"]
    label_list = raw_datasets[data_type]["label"] # [0, 1, 0, 0, ...]
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


class SRLDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args) -> None:
        super().__init__()
        raw_datasets = load_dataset(f'tasks/srl/datasets/{data_args.dataset_name}.py')
        self.tokenizer = tokenizer

        if training_args.do_train:
            column_names = raw_datasets["train"].column_names
            features = raw_datasets["train"].features
        else:
            column_names = raw_datasets["validation"].column_names
            features = raw_datasets["validation"].features

        self.label_column_name = f"tags"
        self.label_list = features[self.label_column_name].feature.names
        self.label_to_id = {l: i for i, l in enumerate(self.label_list)}
        self.num_labels = len(self.label_list)

        if training_args.do_train:
            train_dataset = raw_datasets['train']
            if data_args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(data_args.max_train_samples))
            self.train_dataset = train_dataset.map(
                self.tokenize_and_align_labels,
                batched=True,
                load_from_cache_file=True,
                desc="Running tokenizer on train dataset",
            )
        if training_args.do_eval:
            eval_dataset = raw_datasets['validation']
            if data_args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
            self.eval_dataset = eval_dataset.map(
                self.tokenize_and_align_labels,
                batched=True,
                load_from_cache_file=True,
                desc="Running tokenizer on validation dataset",
            )

        if training_args.do_predict:
            if data_args.dataset_name == "conll2005":
                self.predict_dataset = OrderedDict()
                self.predict_dataset['wsj'] = raw_datasets['test_wsj'].map(
                    self.tokenize_and_align_labels,
                    batched=True,
                    load_from_cache_file=True,
                    desc="Running tokenizer on WSJ test dataset",
                )

                self.predict_dataset['brown'] = raw_datasets['test_brown'].map(
                    self.tokenize_and_align_labels,
                    batched=True,
                    load_from_cache_file=True,
                    desc="Running tokenizer on Brown test dataset",
                )
            else:
                self.predict_dataset = raw_datasets['test_wsj'].map(
                    self.tokenize_and_align_labels,
                    batched=True,
                    load_from_cache_file=True,
                    desc="Running tokenizer on WSJ test dataset",
                )
        
        # add by wjn 
        # 随机采样few-shot training / dev data（传入label_list，对每个label进行采样，最后得到索引列表）
        if data_args.num_examples_per_label is not None:
            train_examples_idx_list = random_sampling(
                raw_datasets=raw_datasets, 
                data_type="train", 
                num_examples_per_label=data_args.num_examples_per_label
            )
            self.train_dataset = self.train_dataset.select(train_examples_idx_list)
            print("Randomly sampling {}-shot training examples for each label. Total examples number is {}".format(
                data_args.num_examples_per_label, 
                len(self.train_dataset)
                ))

        self.data_collator = DataCollatorForTokenClassification(self.tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
        self.metric = load_metric("seqeval")


    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def tokenize_and_align_labels(self, examples):        
        for i, tokens in enumerate(examples['tokens']):
            examples['tokens'][i] = tokens + ["[SEP]"] + [tokens[int(examples['index'][i])]]

        tokenized_inputs = self.tokenizer(
            examples['tokens'],
            padding=False,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        # print(tokenized_inputs['input_ids'][0])

        labels = []
        for i, label in enumerate(examples['tags']):
            word_ids = [None]
            for j, word in enumerate(examples['tokens'][i][:-2]):
                token = self.tokenizer.encode(word, add_special_tokens=False)
                word_ids += [j] * len(token)
            word_ids += [None]
            verb = examples['tokens'][i][int(examples['index'][i])]
            word_ids += [None] * len(self.tokenizer.encode(verb, add_special_tokens=False))
            word_ids += [None]
            
            # word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs