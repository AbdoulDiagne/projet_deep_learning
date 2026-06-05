from __future__ import annotations
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


def load_tokenizer(model_name: str, max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.model_max_length = max_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_policy_model(model_name: str, tokenizer, lora_cfg: dict | None = None):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    if lora_cfg and lora_cfg.get("enabled", False):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules="all-linear",
        )
        model = get_peft_model(model, peft_config)
    return model


def load_reward_model(model_name: str, tokenizer, lora_cfg: dict | None = None):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id
    if lora_cfg and lora_cfg.get("enabled", False):
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules="all-linear",
        )
        model = get_peft_model(model, peft_config)
    return model
