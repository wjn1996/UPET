from datasets.load import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import logging
from collections import defaultdict
from typing import Optional

# add by wjn
def random_sampling(raw_datasets: load_dataset, data_type: str="train", num_examples_per_label: Optional[int]=16):
    assert data_type in ["train", "validation", "test"]
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
    # 保存没有被选中的example id
    num_examples = len(label_list)
    un_selected_examples_ids = [idx for idx in range(num_examples) if idx not in few_example_ids]
    return few_example_ids, un_selected_examples_ids

task_to_test_key = {
    "boolq": "accuracy",
    "cb": "accuracy",
    "rte": "accuracy",
    "wic": "accuracy",
    "wsc": "accuracy",
    "copa": "accuracy",
    "record": "f1",
    "multirc": "f1"
}

task_to_keys = {
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "rte": ("premise", "hypothesis"),
    "wic": ("processed_sentence1", None),
    "wsc": ("span2_word_text", "span1_text"),
    "copa": (None, None),
    "record": (None, None),
    "multirc": ("paragraph", "question_answer")
}

task_to_template = {
    "boolq" : [{"prefix_template": "Question: ", "suffix_template": ""}, {"prefix_template": "Passage: ", "suffix_template": " Answer: <mask> ."}],
    "cb": [None, {"prefix_template": "? <mask> ,", "suffix_template": ""}],
    "rte": [None, {"prefix_template": "? <mask> ,", "suffix_template": ""}], # prefix / suffix template in each segment.
    # "wic": [None, None],
    # "wsc": [None, None],
    # "copa": [None, None],
    # "record": [None, None],
    # "multirc": [None, None],
}

# add by wjn
label_words_mapping = {
    "boolq": {"True": ["Yes"], "False": ["No"]},
    "cb": {"contradiction": ["No"], "neutral": ["Maybe"], "entailment": ["Yes"]},
    "rte": {"not_entailment": ["No"], "entailment": ["Yes"]}
}


logger = logging.getLogger(__name__)


class SuperGlueDataset():
    def __init__(
        self, 
        tokenizer: AutoTokenizer, 
        data_args, 
        training_args, 
        semi_training_args=None,
        use_prompt=False
    ) -> None:
        super().__init__()
        raw_datasets = load_dataset("super_glue", data_args.dataset_name)
        self.tokenizer = tokenizer
        self.data_args = data_args
        
        self.multiple_choice = data_args.dataset_name in ["copa"]

        if data_args.dataset_name == "record":
            self.num_labels = 2
            self.label_list = ["0", "1"]
        elif not self.multiple_choice:
            self.label_list = raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
        else:
            self.num_labels = 1

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
                    assert len(tokenizer.tokenize(' ' + self.label_to_word[key])) == 1
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

        if not self.multiple_choice:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}
            print(f"{self.label2id}")
            print(f"{self.id2label}")

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        if data_args.dataset_name == "record":
            raw_datasets = raw_datasets.map(
                self.record_preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )
        else:
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

        if training_args.do_eval:
            self.eval_dataset = raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            self.predict_dataset = raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))
        

        # add by wjn 
        self.unlabeled_dataset = None
        if semi_training_args.use_semi is True:
            assert data_args.num_examples_per_label is not None and data_args.num_examples_per_label != -1
        
        # 随机采样few-shot training / dev data（传入label_list，对每个label进行采样，最后得到索引列表）
        if data_args.num_examples_per_label is not None and data_args.num_examples_per_label != -1:
            train_examples_idx_list, un_selected_idx_list = random_sampling(
                raw_datasets=raw_datasets, 
                data_type="train", 
                num_examples_per_label=data_args.num_examples_per_label
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

        self.metric = load_metric("./metrics/super_glue", data_args.dataset_name)

        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

        # self.test_key = "accuracy" if data_args.dataset_name not in ["record", "multirc"] else "f1"

        # ==== define test key ====== add by wjn
        self.test_key = task_to_test_key[data_args.dataset_name]

    def preprocess_function(self, examples):

        # print("examples[self.sentence1_key]=", examples[self.sentence1_key][0])

        # WSC
        if self.data_args.dataset_name == "wsc":
            examples["span2_word_text"] = []
            for text, span2_index, span2_word in zip(examples["text"], examples["span2_index"], examples["span2_text"]):
                if self.data_args.template_id == 0:
                    examples["span2_word_text"].append(span2_word + ": " + text)
                elif self.data_args.template_id == 1:
                    words_a = text.split()
                    words_a[span2_index] = "*" + words_a[span2_index] + "*"
                    examples["span2_word_text"].append(' '.join(words_a))

        # WiC
        if self.data_args.dataset_name == "wic":
            examples["processed_sentence1"] = []
            if self.data_args.template_id == 1:
                self.sentence2_key = "processed_sentence2"
                examples["processed_sentence2"] = []
            for sentence1, sentence2, word, start1, end1, start2, end2 in zip(examples["sentence1"], examples["sentence2"], examples["word"], examples["start1"], examples["end1"], examples["start2"], examples["end2"]):
                if self.data_args.template_id == 0: #ROBERTA
                    examples["processed_sentence1"].append(f"{sentence1} {sentence2} Does {word} have the same meaning in both sentences?")
                elif self.data_args.template_id == 1: #BERT
                    examples["processed_sentence1"].append(word + ": " + sentence1)
                    examples["processed_sentence2"].append(word + ": " + sentence2)

        # MultiRC
        if self.data_args.dataset_name == "multirc":
            examples["question_answer"] = []
            for question, answer in zip(examples["question"], examples["answer"]):
                examples["question_answer"].append(f"{question} {answer}")

        # COPA
        if self.data_args.dataset_name == "copa":
            examples["text_a"] = []
            for premise, question in zip(examples["premise"], examples["question"]):
                joiner = "because" if question == "cause" else "so"
                text_a = f"{premise} {joiner}"                    
                examples["text_a"].append(text_a)

            result1 = self.tokenizer(examples["text_a"], examples["choice1"], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
            result2 = self.tokenizer(examples["text_a"], examples["choice2"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
            result = {}  
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                if key in result1 and key in result2:
                    result[key] = []
                    for value1, value2 in zip(result1[key], result2[key]):
                        result[key].append([value1, value2])
            return result

        # add by wjn
        # adding prompt into each example
        if self.use_prompt:
            # if use prompt, insert template into example
            examples = self.prompt_preprocess_function(examples)
        

        args = (
            (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
        )
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

        # add by wjn
        # adding mask pos feature
        
        # print("self.use_prompt=", self.use_prompt)
        if self.use_prompt:
            mask_pos = []
            for input_ids in result["input_ids"]:
                mask_pos.append(input_ids.index(self.special_token_mapping["mask"]))
            result["mask_pos"] = mask_pos
            # print("result=", result)
        # print("*"*100)
        # print("result.keys()=", result.keys())
        # print("result['token_type_ids']=", result["token_type_ids"][0])
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
        preds = np.argmax(preds, axis=1)

        if self.data_args.dataset_name == "record":
            return self.reocrd_compute_metrics(p)

        if self.data_args.dataset_name == "multirc":
            from sklearn.metrics import f1_score
            return {"f1": f1_score(preds, p.label_ids)}

        if self.data_args.dataset_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    def reocrd_compute_metrics(self, p: EvalPrediction):
        from tasks.superglue.utils import f1_score, exact_match_score, metric_max_over_ground_truths
        probs = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        examples = self.eval_dataset
        qid2pred = defaultdict(list)
        qid2ans = {}
        for prob, example in zip(probs, examples):
            qid = example['question_id']
            qid2pred[qid].append((prob[1], example['entity']))
            if qid not in qid2ans:
                qid2ans[qid] = example['answers']
        n_correct, n_total = 0, 0
        f1, em = 0, 0
        for qid in qid2pred:
            preds = sorted(qid2pred[qid], reverse=True)
            entity = preds[0][1]
            n_total += 1
            n_correct += (entity in qid2ans[qid])
            f1 += metric_max_over_ground_truths(f1_score, entity, qid2ans[qid])
            em += metric_max_over_ground_truths(exact_match_score, entity, qid2ans[qid])
        acc = n_correct / n_total
        f1 = f1 / n_total
        em = em / n_total
        return {'f1': f1, 'exact_match': em}

    def record_preprocess_function(self, examples, split="train"):
        results = {
            "index": list(),
            "question_id": list(),
            "input_ids": list(),
            "attention_mask": list(),
            "token_type_ids": list(),
            "label": list(),
            "entity": list(),
            "answers": list()
        }
        for idx, passage in enumerate(examples["passage"]):
            query, entities, answers =  examples["query"][idx], examples["entities"][idx], examples["answers"][idx]
            index = examples["idx"][idx]
            passage = passage.replace("@highlight\n", "- ")
            
            for ent_idx, ent in enumerate(entities):
                question = query.replace("@placeholder", ent)
                result = self.tokenizer(passage, question, padding=self.padding, max_length=self.max_seq_length, truncation=True)
                label = 1 if ent in answers else 0

                results["input_ids"].append(result["input_ids"])
                results["attention_mask"].append(result["attention_mask"])
                if "token_type_ids" in result: results["token_type_ids"].append(result["token_type_ids"])
                results["label"].append(label)
                results["index"].append(index)
                results["question_id"].append(index["query"])
                results["entity"].append(ent)
                results["answers"].append(answers)

        return results
