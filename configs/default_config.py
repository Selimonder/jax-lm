from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# generic
cfg.seed = 1
cfg.fold = 0
cfg.output_dir = "outs/"
cfg.logging_steps = 10
cfg.push_to_hub = False
cfg.use_auth_token = False

# dataset
cfg.input   = "/kaggle/input/feedback-prize-effectiveness/"
cfg.dataset = "default_data"

# model
cfg.model_name_or_path = "bert-base-cased" #"google/bigbird-roberta-base" #"bert-base-cased"
cfg.num_labels = 3
cfg.use_slow_tokenizer = False

# train
cfg.num_train_epochs = 3
cfg.max_seq_length = 256
cfg.max_len = 256
cfg.warmup_steps = 0
cfg.learning_rate = 1e-5
cfg.weight_decay = 0.001 # 0 for now -> 0.001
cfg.per_device_train_batch_size = 8
cfg.per_device_eval_batch_size = 8
cfg.eval_steps = 459
cfg.save_steps = cfg.eval_steps

basic_cfg = cfg