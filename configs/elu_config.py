from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# generic
cfg.seed = 1
cfg.fold = 0
cfg.experiment_suffix = "data-fix"
cfg.output_dir = f"outs/base-{cfg.experiment_suffix}/"
cfg.logging_steps = 10
cfg.push_to_hub = False
cfg.use_auth_token = False
cfg.wandb = True


# dataset
cfg.input   = "/kaggle/input/feedback-prize-effectiveness/"
cfg.dataset = "default_data"

# model
cfg.model = "default_model"
cfg.model_name_or_path = "roberta-base" #"google/bigbird-roberta-base" #"bert-base-cased" # "siebert/sentiment-roberta-large-english" 

## course5i/SEAD-L-6_H-384_A-12-wnli
cfg.num_labels = 3
cfg.use_slow_tokenizer = False

# train
cfg.loss_type = "ce_smooth"
cfg.smoothing_alpha = 0.03
cfg.poly1_epsilon = 1.0
cfg.num_train_epochs = 6
cfg.max_len = 512
cfg.warmup_steps = 0
cfg.optimizer_type = "adamw"
cfg.learning_rate = 3e-5
cfg.weight_decay = 0.01 # 0 for now -> 0.001 # 2e-4 next
cfg.per_device_train_batch_size = 16
cfg.per_device_eval_batch_size = 16
cfg.eval_steps = 430# 430 for bs 8 -> 215 for bs 4
cfg.save_steps = cfg.eval_steps

basic_cfg = cfg