"""Minimal training script using Jax/Flax/HF"""
import os, sys
import argparse

import importlib

from copy import copy

from transformers import (
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForSequenceClassification,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    is_tensorboard_available,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
sys.path.append("configs")
# sys.path.append("models")
# sys.path.append("data")

parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename")
parser.add_argument("-s", "--seed", type=int, default=-1, help="seed")
parser.add_argument("-f", "--fold", type=int, default=-1, help="fold")
parser_args, _ = parser.parse_known_args(sys.argv)
cfg = copy(importlib.import_module(parser_args.config).cfg)


if __name__ == "__main__":
    print(cfg)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        cfg.model_name_or_path,
        num_labels=cfg.num_labels,
        #finetuning_task=data_args.task_name,
        #use_auth_token=True if cfg.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        use_fast=not cfg.use_slow_tokenizer,
        #use_auth_token=True if cfg.use_auth_token else None,
    )
    model = FlaxAutoModelForSequenceClassification.from_pretrained(
        cfg.model_name_or_path,
        config=config,
        #use_auth_token=True if cfg.use_auth_token else None,
    )