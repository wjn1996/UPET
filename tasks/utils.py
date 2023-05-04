from tasks.glue.dataset import task_to_keys as glue_tasks
from tasks.superglue.dataset import task_to_keys as superglue_tasks
from datasets.load import load_dataset, load_metric


GLUE_DATASETS = list(glue_tasks.keys())
SUPERGLUE_DATASETS = list(superglue_tasks.keys())
NER_DATASETS = ["conll2003", "conll2004", "ontonotes"]
SRL_DATASETS = ["conll2005", "conll2012"]
QA_DATASETS = ["squad", "squad_v2"]
OTHER_DATASETS = ["movie_rationales", "cr", "snli", "trec", "ag_news", "yelp_polarity"]


TASKS = ["glue", "superglue", "ner", "srl", "qa", "other_cls"]

DATASETS = GLUE_DATASETS + SUPERGLUE_DATASETS + NER_DATASETS + SRL_DATASETS + QA_DATASETS + OTHER_DATASETS

ADD_PREFIX_SPACE = {
    'bert': False,
    'roberta': True,
    'deberta': True,
    'gpt2': True,
    'deberta-v2': True,
}

USE_FAST = {
    'bert': True,
    'roberta': True,
    'deberta': True,
    'gpt2': True,
    'deberta-v2': False,
}






# def random_sampling(sentences, labels, num):
#     """randomly sample subset of the training pairs"""
#     assert len(sentences) == len(labels)
#     if num > len(labels):
#         assert False, f"you tried to randomly sample {num}, which is more than the total size of the pool {len(labels)}"
#     idxs = np.random.choice(len(labels), size=num, replace=False)
#     selected_sentences = [sentences[i] for i in idxs]
#     selected_labels = [labels[i] for i in idxs]
#     return deepcopy(selected_sentences), deepcopy(selected_labels)