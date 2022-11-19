from torchmetrics import Accuracy, F1Score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from transformers.configuration_utils import PretrainedConfig
from composer.models import HuggingFaceModel
from composer.metrics import CrossEntropy

from mosaic.args import ModelArguments


class ComposerHFModelWithTokenizer(HuggingFaceModel):
    def __init__(self, *args, **kwargs):
        self.tokenizer = kwargs['tokenizer']
        super().__init__(*args, **kwargs)


def get_huggingface_model(model_args: ModelArguments, hf_config: PretrainedConfig):

    metrics = [CrossEntropy(), Accuracy(), F1Score(num_classes=hf_config.num_labels, average="macro")]

    tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        do_lower_case=model_args.do_lower_case,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # todo also allow seq2seq
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=hf_config,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # GPT2 does not have a PAD token, and that breaks things
    # this lets GPT2 be used in this script without errors
    if "gpt" in tokenizer_name or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return ComposerHFModelWithTokenizer(model=model, tokenizer=tokenizer, metrics=metrics, use_logits=True)
    
