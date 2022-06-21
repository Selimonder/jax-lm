from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# generic
cfg.seed = 1
cfg.fold = 0
cfg.output_dir = "outs/roberta-large-bigbird_01_poly1_ce_labelsmoothing_weightdecay/"
cfg.logging_steps = 10
cfg.push_to_hub = False
cfg.use_auth_token = False

# dataset
cfg.input   = "/kaggle/input/feedback-prize-effectiveness/"
cfg.dataset = "default_data"

# model
cfg.model = "default_model"
cfg.model_name_or_path = "google/bigbird-roberta-large" #"google/bigbird-roberta-base" #"bert-base-cased" # "siebert/sentiment-roberta-large-english" 

## course5i/SEAD-L-6_H-384_A-12-wnli
cfg.num_labels = 3
cfg.use_slow_tokenizer = False

# train
cfg.loss_type = "poly1_ce_ls"
cfg.poly1_epsilon = 1.0
cfg.num_train_epochs = 21
cfg.max_len = 1024
cfg.warmup_steps = 0
cfg.learning_rate = 1e-5
cfg.weight_decay = 2e-4 # 0 for now -> 0.001 # 2e-4 next
cfg.per_device_train_batch_size = 1
cfg.per_device_eval_batch_size = 2
cfg.eval_steps = 3676 # 459 for bs 8 -> 919 for bs 4
cfg.save_steps = cfg.eval_steps

basic_cfg = cfg