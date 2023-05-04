# UPET: Uncertainty-aware Parameter-Efficient Tuning for Semi-supervised Language Understanding


Head Tuning: Training the model with CLS head, whith or whitout prefix / adapter
Prompt Tuning: Training the model with prompt and verbalizer (MLM head), whith or whitout prefix / adapter

augment definition：
e.g.,
--prefix -> --head-prefix or --prompt-prefix
--prompt -> --head-ptuning or --prompt-ptuning

### Setup
We conduct our experiment with Anaconda3. If you have installed Anaconda3, then create the environment for P-tuning v2:

```shell
conda create -n pt2 python=3.8.5
conda activate pt2
```

After we setup basic conda environment, install pytorch related packages via:

```shell
conda install -n pt2 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

Finally, install other python packages we need:

```shell
pip install -r requirements.txt
```

### Data
For SuperGLUE and SQuAD datasets, we download them from the Huggingface Datasets APIs (embedded in our codes).

For sequence tagging (NER, SRL) datasets, we prepare a non-official packup [here](https://zenodo.org/record/6318701/files/P-tuning-v2_data.tar.gz?download=1). 
After downloading, unzip the packup to the project root.
Please use at your own risk.

### Training
Run training scripts in [run_script](run_script) (e.g., RoBERTa for RTE):

You can change the augments and run:
```shell
bash run_script/run_rte_roberta.sh
```
or

```shell
export TASK_NAME=superglue
export DATASET_NAME=rte
export CUDA_VISIBLE_DEVICES=0
bs=32
lr=5e-3
dropout=0.1
psl=128
epoch=100
python3 run.py \
  --model_name_or_path /wjn/pre-trained-lm/roberta-large \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --output_dir checkpoints/$DATASET_NAME-roberta/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix
```
This script is run for Full-data Full-supervised Pre-fix Tuning.

We provide the following kinds of settings:
- Full-data v.s. Few-shot: The training data is full / few-shot
- Full-supervised v.s. Semi-supervised: We use full-supervised / self-training
- Full-Tuning v.s. Patameter-efficient Tuning: Only tuning the full parameters / Tuning the few parameters
- One-stage v.s. Two-stage: directly tuning / tuning the few paraemters and then tuning the full
- Head-Tuning v.s. Prompt-Tuning: Prefix/Adapter + CLS head / Prefix/Adapter + Prompt + Vaberlizer

The specific augments for different scenarios:


#### Few-shot Head Tuning

### Implemented Results
Currently we have released our reimplementation on following tasks and datasets. More implementation will be released soon.

Released results on BERT-large

|              | BoolQ | COPA | RTE  | WiC  | WSC  | CoNLL04 | OntoNotes 5.0 | CoNLL12 |
|--------------|-------|------|------|------|------|---------|---------------|---------|
| Result       | 74.3  | 77.0 | 80.1 | 75.1 | 68.3 | 84.5    | 86.4          | 85.3    |
| Total Epochs | 100   | 80   | 60   | 80   | 80   | 40      | 30            | 45      |
| Best Epoch   | 58    | 12   | 30   | 56   | 17   | 33      | 24            | 43      |

Released results on RoBERTa-large

|              | BoolQ | COPA | RTE  | WiC  | WSC  | CoNLL03 | CoNLL04 | OntoNotes 5.0 | CoNLL12 | CoNLL05 WSJ | CoNLL05 Brown | SQuAD 1.1 | SQuAD 2.0 |
|--------------|-------|------|------|------|------|---------|---------|---------------|---------|-------------|---------------|-----------|-----------|
| Results      | 84.0  | 92.0 | 86.6 | 73.7 | 64.4 | 91.8    | 88.4    | 90.1          | 84.7    | 89.4        | 83.9          | 88.1/94.2 | 81.3/84.7 |
| Total Epochs | 100   | 120  | 100  | 50   | 10   | 30      | 80      | 60            | 45      | 15          | -             | 30        | 10        |
| Best Epoch   | 86    | 78   | 65   | 31   | 3    | 28      | 45      | 59            | 37      | 13          | -             | 24        | 9         |

For other hyper-parameters, please refer to the training scripts. 
If you can not achieve the reported results at the best epoch, there is probably an environmental mismatch and hyper-parameter search is needed.

## Citation

If you find our work useful, please kindly cite our paper:






---

use_prompt
sst-2,mr,cr,mnli,snli,qnli,rte,mrpc,qqp,cola,trec



superglue
BoolQ	CB	COPA	MultiRC	ReCoRD	RTE	WiC	WSC	AX-b	AX-g
glue
CoLA	SST-2	MRPC	STS-B	QQP	MNLI-m	MNLI-mm	QNLI	RTE	WNLI	AX


