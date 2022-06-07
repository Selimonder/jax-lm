"""Minimal training script using Jax/Flax/HF"""
import os, sys, time
import argparse
import logging
import importlib

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

from tqdm import tqdm
from copy import copy
from transformers.utils import check_min_version, get_full_repo_name

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

def create_learning_rate_fn(
    train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn

def create_train_state(
    model: FlaxAutoModelForSequenceClassification,
    learning_rate_fn: Callable[[int], float],
    num_labels: int,
    weight_decay: float,
) -> train_state.TrainState:
    """Create initial training state."""

    class TrainState(train_state.TrainState):
        """Train state with an Optax optimizer.

        The two functions below differ depending on whether the task is classification
        or regression.

        Args:
          logits_fn: Applied to last layer to obtain the logits.
          loss_fn: Function to compute the loss.
        """

        logits_fn: Callable = struct.field(pytree_node=False)
        loss_fn: Callable = struct.field(pytree_node=False)

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    tx = optax.adamw(
        learning_rate=learning_rate_fn, b1=0.9, b2=0.999, eps=1e-6, weight_decay=weight_decay, mask=decay_mask_fn
    )

    def cross_entropy_loss(logits, labels):
        xentropy = optax.softmax_cross_entropy(logits, onehot(labels, num_classes=num_labels))
        return jnp.mean(xentropy)

    return TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=tx,
        logits_fn=lambda logits: logits.argmax(-1),
        loss_fn=cross_entropy_loss,
    )


# define step functions
def train_step(
    state: train_state.TrainState, batch: Dict[str, Array], dropout_rng: PRNGKey
) -> Tuple[train_state.TrainState, float]:
    """Trains model with an optimizer (both in `state`) on `batch`, returning a pair `(new_state, loss)`."""
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
    targets = batch.pop("labels")

    def loss_fn(params):
        logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
        loss = state.loss_fn(logits, targets)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)
    metrics = jax.lax.pmean({"loss": loss, "learning_rate": learning_rate_fn(state.step)}, axis_name="batch")
    return new_state, metrics, new_dropout_rng

def eval_step(state, batch):
    logits = state.apply_fn(**batch, params=state.params, train=False)[0]
    return state.logits_fn(logits)


def train_data_collator(rng: PRNGKey, dataset: Dataset, batch_size: int):
    """Returns shuffled batches of size `batch_size` from truncated `train dataset`, sharded over all local devices."""
    steps_per_epoch = len(dataset) // batch_size
    perms = jax.random.permutation(rng, len(dataset))
    perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in perms:
        batch = dataset[perm]
        discourse_id, input_ids, labels = dataset[perm]['discourse_id'], dataset[perm]['input_ids'], dataset[perm]['label']
        batch.pop("discourse_id", None)
        batch = {"input_ids": np.array(input_ids), "mask": [np.ones_like(x) for x in input_ids], "labels": np.array(labels)}

        # calculate max token length of this batch
        batch_max = 512 #max([len(ids) for ids in batch["input_ids"]])
        # add padding
        if tokenizer.padding_side == "right":
            batch["input_ids"] = [s + (batch_max - len(s)) * [tokenizer.pad_token_id] for s in batch["input_ids"]]
        else:
            batch["input_ids"] = [(batch_max - len(s)) * [tokenizer.pad_token_id] + s for s in batch["input_ids"]]

        batch['input_ids'] = [x[:batch_max] for x in batch['input_ids']]

        batch['input_ids'] = np.stack(batch['input_ids'])
        
        masks = np.zeros_like(batch['input_ids'])
        masks[batch['input_ids'] != tokenizer.pad_token_id] = 1
        batch['mask'] = masks

        batch.pop("mask", None)

        # batch = {k: np.array(v) for k, v in batch.items()}
        batch = shard(batch)
        yield batch

def eval_data_collator(dataset: Dataset, batch_size: int):
    """Returns batches of size `batch_size` from `eval dataset`, sharded over all local devices."""
    for i in range(len(dataset) // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size]
        discourse_id, input_ids, labels = dataset[i * batch_size : (i + 1) * batch_size]['discourse_id'], dataset[i * batch_size : (i + 1) * batch_size]['input_ids'], dataset[i * batch_size : (i + 1) * batch_size]['label']
        batch.pop("discourse_id", None)
        batch = {"input_ids": np.array(input_ids), "mask": [np.ones_like(x) for x in input_ids], "labels": np.array(labels)}

        # calculate max token length of this batch
        batch_max = 512 #max([len(ids) for ids in batch["input_ids"]])
        # add padding
        if tokenizer.padding_side == "right":
            batch["input_ids"] = [s + (batch_max - len(s)) * [tokenizer.pad_token_id] for s in batch["input_ids"]]
        else:
            batch["input_ids"] = [(batch_max - len(s)) * [tokenizer.pad_token_id] + s for s in batch["input_ids"]]

        batch['input_ids'] = [x[:batch_max] for x in batch['input_ids']]

        batch['input_ids'] = np.stack(batch['input_ids'])
        
        masks = np.zeros_like(batch['input_ids'])
        masks[batch['input_ids'] != tokenizer.pad_token_id] = 1
        batch['mask'] = masks
        batch.pop("mask", None)

        # batch = {k: np.array(v) for k, v in batch.items()}
        batch = shard(batch)

        yield batch

    
if __name__ == "__main__":
    
    # Logger
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

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

    # TODO: check overlapping data
    # Choose a fold
    logger.info(f"creating dataset for fold {cfg.fold}")
    train_dataset = load_dataset("json", data_files=f"/kaggle/working/folds/train_{cfg.fold}.jsonl", split="train")
    eval_dataset   = load_dataset("json", data_files=f"/kaggle/working/folds/valid_{cfg.fold}.jsonl", split="train")

    num_epochs = int(cfg.num_train_epochs)
    rng = jax.random.PRNGKey(cfg.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())

    train_batch_size = cfg.per_device_train_batch_size * jax.local_device_count()
    eval_batch_size = cfg.per_device_eval_batch_size * jax.local_device_count()

    learning_rate_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        cfg.num_train_epochs,
        cfg.warmup_steps,
        cfg.learning_rate,
    )

    state = create_train_state(
        model, learning_rate_fn, num_labels=cfg.num_labels, weight_decay=cfg.weight_decay
    )

    p_train_step = jax.pmap(train_step, axis_name="batch", donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, axis_name="batch")
    
    # TODO: competition metric
    metric = load_metric("accuracy")

    logger.info(f"===== Starting training ({num_epochs} epochs) =====")
    train_time = 0

    # make sure weights are replicated on each device
    state = replicate(state)

    steps_per_epoch = len(train_dataset) // train_batch_size
    total_steps = steps_per_epoch * num_epochs
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (0/{num_epochs})", position=0)

    for epoch in epochs:

        train_start = time.time()
        train_metrics = []

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)
        logger.info(f"===== Starting training ({rng}) =====")

        # train
        train_loader = train_data_collator(input_rng, train_dataset, train_batch_size)
        for step, batch in enumerate(
            tqdm(
                train_loader,
                total=steps_per_epoch,
                desc="Training...",
                position=1,
            ),
        ):
            state, train_metric, dropout_rngs = p_train_step(state, batch, dropout_rngs)
            
            train_metrics.append(train_metric)

            train_metric = unreplicate(train_metric)
            cur_step = (epoch * steps_per_epoch) + (step + 1)
            if jax.process_index() == 0:
                epochs.write(
                    f"Step... ({cur_step}/{total_steps} | Training Loss: {train_metric['loss']}, Learning Rate:"
                    f" {train_metric['learning_rate']})"
                )
