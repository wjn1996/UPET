from enum import Enum

from model.head_for_token_classification import (
    BertPrefixForTokenClassification,
    BertAdapterForTokenClassification,
    RobertaPrefixForTokenClassification,
    RobertaAdapterForTokenClassification,
    DebertaPrefixForTokenClassification,
    DebertaV2PrefixForTokenClassification
)

from model.head_for_sequence_classification import (
    BertPrefixForSequenceClassification,
    BertPtuningForSequenceClassification,
    BertAdapterForSequenceClassification,
    RobertaPrefixForSequenceClassification,
    RobertaPtuningForSequenceClassification,
    RobertaAdapterForSequenceClassification,
    DebertaPrefixForSequenceClassification
)

from model.head_for_question_answering import (
    BertPrefixForQuestionAnswering,
    BertAdapterForQuestionAnswering,
    RobertaPrefixModelForQuestionAnswering,
    RobertaAdapterForQuestionAnswering,
    DebertaPrefixModelForQuestionAnswering
)

from model.prompt_for_sequence_classification import (
    PromptBertForSequenceClassification,
    PromptBertPrefixForSequenceClassification,
    PromptBertPtuningForSequenceClassification,
    PromptBertAdapterForSequenceClassification,
    PromptRobertaForSequenceClassification,
    PromptRobertaPrefixForSequenceClassification,
    PromptRobertaPtuningForSequenceClassification,
    PromptRobertaAdapterForSequenceClassification,
    PromptDebertaForSequenceClassification,
    PromptDebertaPrefixForSequenceClassification,
    PromptDebertaPtuningForSequenceClassification,
    # PromptDebertaAdapterForSequenceClassification,
    PromptDebertav2ForSequenceClassification,
    PromptDebertav2PrefixForSequenceClassification,
    PromptDebertav2PtuningForSequenceClassification,
    # PromptDebertav2AdapterForSequenceClassification
)

from model.multiple_choice import (
    BertPrefixForMultipleChoice,
    RobertaPrefixForMultipleChoice,
    DebertaPrefixForMultipleChoice,
    BertPromptForMultipleChoice,
    RobertaPromptForMultipleChoice
)

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice
)


class TaskType(Enum):
    TOKEN_CLASSIFICATION = 1,
    SEQUENCE_CLASSIFICATION = 2,
    QUESTION_ANSWERING = 3,
    MULTIPLE_CHOICE = 4

# used for head fine-tuning
AUTO_MODELS = {
    TaskType.TOKEN_CLASSIFICATION: AutoModelForTokenClassification,
    TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
    TaskType.QUESTION_ANSWERING: AutoModelForQuestionAnswering,
    TaskType.MULTIPLE_CHOICE: AutoModelForMultipleChoice,
}

# used for head prefix-tuning
HEAD_PREFIX_MODELS = {
    "bert": {
        TaskType.TOKEN_CLASSIFICATION: BertPrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: BertPrefixForSequenceClassification,
        TaskType.QUESTION_ANSWERING: BertPrefixForQuestionAnswering,
        TaskType.MULTIPLE_CHOICE: BertPrefixForMultipleChoice
    },
    "roberta": {
        TaskType.TOKEN_CLASSIFICATION: RobertaPrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: RobertaPrefixForSequenceClassification,
        TaskType.QUESTION_ANSWERING: RobertaPrefixModelForQuestionAnswering,
        TaskType.MULTIPLE_CHOICE: RobertaPrefixForMultipleChoice,
    },
    "deberta": {
        TaskType.TOKEN_CLASSIFICATION: DebertaPrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: DebertaPrefixForSequenceClassification,
        TaskType.QUESTION_ANSWERING: DebertaPrefixModelForQuestionAnswering,
        TaskType.MULTIPLE_CHOICE: DebertaPrefixForMultipleChoice,
    },
    "deberta-v2": {
        TaskType.TOKEN_CLASSIFICATION: DebertaV2PrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: None,
        TaskType.QUESTION_ANSWERING: None,
        TaskType.MULTIPLE_CHOICE: None,
    }
}

# used for head p-tuning
HEAD_PTUNING_MODELS = {
    "bert": {
        TaskType.SEQUENCE_CLASSIFICATION: BertPtuningForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: BertPromptForMultipleChoice
    },
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: RobertaPtuningForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: RobertaPromptForMultipleChoice
    }
}

## add by wjn
# used for head adapter-tuning
HEAD_ADAPTER_MODELS = {
    "bert": {
        TaskType.TOKEN_CLASSIFICATION: BertAdapterForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: BertAdapterForSequenceClassification,
        TaskType.QUESTION_ANSWERING: BertAdapterForQuestionAnswering,
        # TaskType.MULTIPLE_CHOICE: BertPrefixForMultipleChoice
    },
    "roberta": {
        TaskType.TOKEN_CLASSIFICATION: RobertaAdapterForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: RobertaAdapterForSequenceClassification,
        TaskType.QUESTION_ANSWERING: RobertaAdapterForQuestionAnswering,
        # TaskType.MULTIPLE_CHOICE: RobertaPrefixForMultipleChoice,
    },
}

# used for prompt prefix-tuning
PROMPT_PREFIX_MODELS = {
    "bert": {
        TaskType.SEQUENCE_CLASSIFICATION: PromptBertPrefixForSequenceClassification,
    },
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: PromptRobertaPrefixForSequenceClassification,
    },
    "deberta": {
         TaskType.SEQUENCE_CLASSIFICATION: PromptDebertaPrefixForSequenceClassification,
    },
     "deberta-v2": {
         TaskType.SEQUENCE_CLASSIFICATION: PromptDebertav2PrefixForSequenceClassification,
    },
}


# used for prompt only
PROMPT_ONLY_MODELS = {
    "bert": {
        TaskType.SEQUENCE_CLASSIFICATION: PromptBertForSequenceClassification,
    },
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: PromptRobertaForSequenceClassification,
    },
    "deberta": {
         TaskType.SEQUENCE_CLASSIFICATION: PromptDebertaForSequenceClassification,
    },
     "deberta-v2": {
         TaskType.SEQUENCE_CLASSIFICATION: PromptDebertav2ForSequenceClassification,
    },
}

# used for prompt p-tuning
PROMPT_PTUNING_MODELS = {
    "bert": {
        TaskType.SEQUENCE_CLASSIFICATION: PromptBertPtuningForSequenceClassification,
    },
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: PromptRobertaPtuningForSequenceClassification,
    },
    "deberta": {
         TaskType.SEQUENCE_CLASSIFICATION: PromptDebertaPtuningForSequenceClassification,
    },
     "deberta-v2": {
         TaskType.SEQUENCE_CLASSIFICATION: PromptDebertav2PtuningForSequenceClassification,
    },
}

# used for prompt adapter
PROMPT_ADAPTER_MODELS = {
    "bert": {
        TaskType.SEQUENCE_CLASSIFICATION: PromptBertAdapterForSequenceClassification,
    },
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: PromptRobertaAdapterForSequenceClassification,
    },
    # "deberta": {
    #      TaskType.SEQUENCE_CLASSIFICATION: PromptDebertaAdapterForSequenceClassification,
    # },
    #  "deberta-v2": {
    #      TaskType.SEQUENCE_CLASSIFICATION: PromptDebertav2AdapterForSequenceClassification,
    # },
}



def get_model(data_args, model_args, task_type: TaskType, config: AutoConfig, fix_bert: bool = False):
    
    
    # whether to fixed parameters of backbone (use pe)
    config.use_pe = model_args.use_pe

    if model_args.head_only:
        model_class = AUTO_MODELS[task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    elif model_args.head_prefix:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection
        config.prefix_hidden_size = model_args.prefix_hidden_size
        
        model_class = HEAD_PREFIX_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    elif model_args.head_ptuning:
        config.pre_seq_len = model_args.pre_seq_len
        model_class = HEAD_PTUNING_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    elif model_args.head_adapter:
        config.adapter_choice = model_args.adapter_choice
        config.adapter_dim = model_args.adapter_dim
        model_class = HEAD_ADAPTER_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    elif model_args.prompt_only:
        config.pre_seq_len = 0
        model_class = PROMPT_ONLY_MODELS[config.model_type][task_type]
        model = model_class(config, model_args, data_args)

    elif model_args.prompt_prefix:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection
        config.prefix_hidden_size = model_args.prefix_hidden_size
        model_class = PROMPT_PREFIX_MODELS[config.model_type][task_type]
        model = model_class(config, model_args, data_args)
        # model = model_class.from_pretrained(
        #     model_args.model_name_or_path,
        #     config=config,
        #     revision=model_args.model_revision,
        # )
    elif model_args.prompt_ptuning:
        config.pre_seq_len = model_args.pre_seq_len
        model_class = PROMPT_PTUNING_MODELS[config.model_type][task_type]
        model = model_class(config, model_args, data_args)
        # model = model_class.from_pretrained(
        #     model_args.model_name_or_path,
        #     config=config,
        #     revision=model_args.model_revision,
        # )
    elif model_args.prompt_adapter:
        config.adapter_choice = model_args.adapter_choice
        config.adapter_dim = model_args.adapter_dim
        model_class = PROMPT_ADAPTER_MODELS[config.model_type][task_type]
        model = model_class(config, model_args, data_args)
        # model = model_class.from_pretrained(
        #     model_args.model_name_or_path,
        #     config=config,
        #     revision=model_args.model_revision,
        # )


    else:
        model_class = AUTO_MODELS[task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )

    # bert_param = 0
    if model_args.head_only and model_args.use_pe:
        if config.model_type == "bert":
            for param in model.bert.parameters():
                param.requires_grad = False
            # for _, param in model.bert.named_parameters():
            #     bert_param += param.numel()
        elif config.model_type == "roberta":
            for param in model.roberta.parameters():
                param.requires_grad = False
            # for _, param in model.roberta.named_parameters():
            #     bert_param += param.numel()
        elif config.model_type == "deberta":
            for param in model.deberta.parameters():
                param.requires_grad = False
            # for _, param in model.deberta.named_parameters():
            #     bert_param += param.numel()
    all_param = 0
    for _, param in model.named_parameters():
        if param.requires_grad is True:
            all_param += param.numel()
    total_param = all_param
    print('***** total param is {} *****'.format(total_param))
    return model


def get_model_deprecated(model_args, task_type: TaskType, config: AutoConfig, fix_bert: bool = False):
    if model_args.prefix:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection
        config.prefix_hidden_size = model_args.prefix_hidden_size

        if task_type == TaskType.TOKEN_CLASSIFICATION:
            from model.head_for_token_classification import BertPrefixModel, RobertaPrefixModel, DebertaPrefixModel, DebertaV2PrefixModel
        elif task_type == TaskType.SEQUENCE_CLASSIFICATION:
            from model.head_for_sequence_classification import BertPrefixModel, RobertaPrefixModel, DebertaPrefixModel, DebertaV2PrefixModel
        elif task_type == TaskType.QUESTION_ANSWERING:
            from model.head_for_question_answering import BertPrefixModel, RobertaPrefixModel, DebertaPrefixModel, DebertaV2PrefixModel
        elif task_type == TaskType.MULTIPLE_CHOICE:
            from model.multiple_choice import BertPrefixModel

        if config.model_type == "bert":
            model = BertPrefixModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif config.model_type == "roberta":
            model = RobertaPrefixModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif config.model_type == "deberta":
            model = DebertaPrefixModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif config.model_type == "deberta-v2":
            model = DebertaV2PrefixModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        else:
            raise NotImplementedError


    elif model_args.prompt:
        config.pre_seq_len = model_args.pre_seq_len

        from model.head_for_sequence_classification import BertPromptModel, RobertaPromptModel
        if config.model_type == "bert":
            model = BertPromptModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif config.model_type == "roberta":
            model = RobertaPromptModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        else:
            raise NotImplementedError
            

    else:
        if task_type == TaskType.TOKEN_CLASSIFICATION:
            model = AutoModelForTokenClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
            
        elif task_type == TaskType.SEQUENCE_CLASSIFICATION:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )

        elif task_type == TaskType.QUESTION_ANSWERING:
            model = AutoModelForQuestionAnswering.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif task_type == TaskType.MULTIPLE_CHOICE:
            model = AutoModelForMultipleChoice.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
    
        bert_param = 0
        if fix_bert:
            if config.model_type == "bert":
                for param in model.bert.parameters():
                    param.requires_grad = False
                for _, param in model.bert.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "roberta":
                for param in model.roberta.parameters():
                    param.requires_grad = False
                for _, param in model.roberta.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "deberta":
                for param in model.deberta.parameters():
                    param.requires_grad = False
                for _, param in model.deberta.named_parameters():
                    bert_param += param.numel()
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('***** total param is {} *****'.format(total_param))
    return model


