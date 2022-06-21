from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# generic
cfg.seed = 1
cfg.fold = 0
cfg.output_dir = "outs/bert2"
cfg.logging_steps = 10
cfg.push_to_hub = False
cfg.use_auth_token = False

# dataset
cfg.input   = "/kaggle/input/feedback-prize-effectiveness/"
cfg.dataset = "default_data"

# model
cfg.model_name_or_path = "google/bigbird-roberta-large" #"google/bigbird-roberta-base" #"bert-base-cased"
cfg.num_labels = 3
cfg.use_slow_tokenizer = False

# train
cfg.num_train_epochs = 5
cfg.max_seq_length = 1536
cfg.max_len = 1536
cfg.warmup_steps = 0
cfg.learning_rate = 2e-5
cfg.weight_decay = 0. # 0 for now -> 0.001
cfg.per_device_train_batch_size = 4
cfg.per_device_eval_batch_size = 4
cfg.eval_steps = 1000 #459
cfg.save_steps = cfg.eval_steps

basic_cfg = cfg