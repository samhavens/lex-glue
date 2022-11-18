from torchmetrics import Accuracy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from composer.models import HuggingFaceModel
from composer.metrics import CrossEntropy

from mosaic.args import ModelArgs


def get_huggingface_model(model_args: ModelArgs, hf_config):

    metrics = [CrossEntropy(), Accuracy()]

    tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=hf_config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    return HuggingFaceModel(model=model, tokenizer=tokenizer, metrics=metrics, use_logits=True)
    
