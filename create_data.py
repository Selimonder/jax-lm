"""Minimal training script using Jax/Flax/HF"""
import os, sys, time, json
import argparse
import logging
import importlib
from joblib import Parallel, delayed

from typing import Any, Callable, Dict, Optional, Tuple

import datasets
from datasets import load_dataset, load_metric

import jax
import jax.numpy as jnp
import optax

from flax import struct, traverse_util
from flax.jax_utils import replicate, unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForSequenceClassification,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    is_tensorboard_available,
)

import numpy as np
import pandas as pd

from tqdm import tqdm
from copy import copy
from transformers.utils import check_min_version, get_full_repo_name

from itertools import chain

Array = Any
Dataset = datasets.arrow_dataset.Dataset
PRNGKey = Any

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.20.0.dev0")

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

LABEL_MAPPING = {"Ineffective": 0, "Adequate": 1, "Effective": 2}

def _prepare_training_data_helper(args, tokenizer, df, is_train):
    training_samples = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        idx = row["essay_id"]
        discourse_text = row["discourse_text"]
        discourse_type = row["discourse_type"]

        if is_train:
            filename = os.path.join(args.input, "train", idx + ".txt")
        else:
            filename = os.path.join(args.input, "test", idx + ".txt")

        with open(filename, "r") as f:
            text = f.read()

        encoded_text = tokenizer.encode_plus(
            discourse_type + " " + discourse_text,
            text,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=1024 ##TODO: update max_length
        )
        input_ids = encoded_text["input_ids"]

        sample = {
            "discourse_id": row["discourse_id"],
            "input_ids": input_ids,
            # "discourse_text": discourse_text,
            # "essay_text": text,
            "attention_mask": encoded_text["attention_mask"],
        }

        if "token_type_ids" in encoded_text:
            sample["token_type_ids"] = encoded_text["token_type_ids"]

        try:
            label = row["discourse_effectiveness"]
            sample["labels"] = LABEL_MAPPING[label]
        except:
            sample["labels"] = 0
        

        training_samples.append(sample)
    return training_samples


def prepare_training_data(df, tokenizer, args, num_jobs, is_train):
    training_samples = []

    df_splits = np.array_split(df, num_jobs)

    results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(_prepare_training_data_helper)(args, tokenizer, df, is_train) for df in df_splits
    )
    for result in results:
        training_samples.extend(result)

    return training_samples

if __name__ == "__main__":
    ## generate test dataset
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
    test = pd.read_csv("/kaggle/input/feedback-prize-effectiveness/test.csv")
    test_data = prepare_training_data(test, tokenizer, cfg, num_jobs=4, is_train=False)
    df = pd.DataFrame.from_records(test_data)
    df.to_json(f"/kaggle/working/folds/test.jsonl", orient="records", lines=True)