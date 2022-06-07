from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# dataset
cfg.input   = "/kaggle/input/feedback-prize-effectiveness/"
cfg.dataset = "default_data"

# model
cfg.model_name_or_path = "google/bigbird-roberta-base"
cfg.num_labels = 3
cfg.use_slow_tokenizer = False

cfg.max_seq_length = 128
cfg.learning_rate = 2e-5
cfg.num_train_epochs = 3
cfg.per_device_train_batch_size = 4
cfg.eval_steps = 100

# generic
cfg.output_dir = "outs/"
cfg.push_to_hub = False
cfg.use_auth_token = False

basic_cfg = cfg