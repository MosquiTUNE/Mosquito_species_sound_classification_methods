# Fine‑tuning AST from SSL Pretrain — README

Fine‑tune (or continue training) an **Audio Spectrogram Transformer (AST)** classifier starting from a **self‑supervised (SSL) pretrained** checkpoint. The script builds train/validation datasets from folders of 1‑second 16 kHz `.wav` clips, attaches a classification head, optionally freezes parts of the encoder, and trains with Hugging Face `Trainer`.

---

## 1) What this notebook does

- Loads an SSL‑pretrained AST (projector/decoder removed) and adds a **linear classifier head**.
- Builds a label map from training subfolders and prepares Hugging Face `Dataset`s.
- Preprocesses audio (16 kHz, trim/pad to 1 s) via `AutoProcessor`.
- Supports **partial fine‑tuning**: last *N* transformer layers, classifier‑only, or full model.
- Uses `Trainer` with **cosine LR**, **mixed precision**, **gradient accumulation**, and a custom **dynamic batch size callback**.
- Saves the best model to an easy‑to‑consume folder under `ensemble/best/`.

---

## 2) Data layout

```
../dataset/train-val-random/
├── train/
│   ├── Class_A/
│   │   ├── *.wav
│   └── Class_B/
│       └── *.wav
└── validation/
    ├── Class_A/
    │   └── *.wav
    └── Class_B/
        └── *.wav
```

Each subfolder name = **class label**. The label map is created from `train/`:

```python
label_list = sorted(os.listdir(train_dir))
label2id = {name: i for i, name in enumerate(label_list)}
id2label = {i: name for name, i in label2id.items()}
```

---

## 3) Key configurable inputs (first cell)

```python
epoch_num = 50
batch_size = 1
batch_sizes = [batch_size] * epoch_num  # can vary per epoch via callback

SSL_pretrained_path = "./SSL_pretrained/AST-SSL-results_SSL_v2_db2_indepval/checkpoint-33297/"

train_dir = "../dataset/train-val-random/train/"
validation_dir = "../dataset/train-val-random/validation/"

postfix = "_AST_from_SSL_25_02_12-3layers"
b_train_just_last = False      # train only the classifier head
trainNlayers = 3               # fine-tune last N encoder layers (0 = full network)

b_just_test = False            # quick debug: downsample dataset if True
random_seed = 42

output_path = "ensemble/"
```

- **`SSL_pretrained_path`**: directory containing your SSL weights (expects `model.safetensors`).
- **`b_train_just_last`** / **`trainNlayers`**: freezing strategy (see §6).
- **`b_just_test`**: if `True`, uses every 10th sample to speed up a dry‑run.
- **`postfix`**: suffix used in log/model dir names.

---

## 4) Reproducibility

```python
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 5) Dataset builder & preprocessing

**Folder → Dataset**
```python
dataset = load_audio_dataset_from_folders(train_dir, validation_dir, b_just_test=b_just_test)
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
```

**Audio → model inputs**
```python
def preprocess_audio_function(examples, target_length=16000):
    # loads 16 kHz, pads/truncates to 1 s, runs processor -> returns tensors
    ...

encoded_train_dataset = train_dataset.map(preprocess_audio_function, batched=True,
                                          remove_columns=train_dataset.column_names)
encoded_validation_dataset = validation_dataset.map(preprocess_audio_function, batched=True,
                                                    remove_columns=train_dataset.column_names)
```

> The processor is `AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")`.

---

## 6) Model assembly (SSL → classifier) & freezing strategy

**Load SSL encoder** and strip projector/decoder:
```python
base_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
class AST_SSL(nn.Module):
    ...  # wraps base_model; we will remove projector/decoder

ssl_model = AST_SSL(base_model, output_dim=[1024, 128])
state_dict = load_file(f"{SSL_pretrained_path}/model.safetensors")
ssl_model.load_state_dict(state_dict)

del ssl_model.projector
verbatim_del = ssl_model.__dict__.get('decoder', None)
del ssl_model.decoder
```

**Classifier head**:
```python
class AST_Classifier(nn.Module):
    def __init__(self, ssl_model, num_classes):
        super().__init__()
        self.encoder = ssl_model.encoder
        self.classifier = nn.Linear(768, num_classes)
    def forward(self, input_values, labels=None):
        x = self.encoder(input_values).last_hidden_state
        x = self.encoder.layernorm(x)
        x = x.mean(dim=1)  # global average pooling over time
        logits = self.classifier(x)
        return {"logits": logits} if labels is None else {"loss": F.cross_entropy(logits, labels), "logits": logits}

model = AST_Classifier(ssl_model, num_classes=len(id2label))
```

**Freezing options**:
```python
if b_train_just_last:
    for p in model.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True
elif trainNlayers > 0:
    for p in model.parameters(): p.requires_grad = False
    for layer in model.encoder.encoder.layer[-trainNlayers:]:
        for p in layer.parameters(): p.requires_grad = True
else:
    for p in model.parameters(): p.requires_grad = True
```

---

## 7) Metrics

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
        "f1": f1_score(labels, preds, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
    }
```

---

## 8) Dynamic batch sizes per epoch

A small `TrainerCallback` reads the `batch_sizes` list and updates train/eval batch sizes **at each epoch start**, also logging to a text file:

```python
class DynamicBatchSizeCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        new_bs = batch_sizes[int(state.epoch)]
        args.per_device_train_batch_size = new_bs
        args.per_device_eval_batch_size = new_bs
        with open(LOG_FILE, "a") as f:
            f.write(f"Epoch {int(state.epoch)+1}/{epoch_num} - New batch size: {new_bs}\n")
```

Initialize with the callback:
```python
callbacks=[DynamicBatchSizeCallback()]
```

---

## 9) Training configuration

```python
training_args = TrainingArguments(
    output_dir = output_path + "./results" + postfix,
    evaluation_strategy = "epoch",
    learning_rate = 5e-5,
    lr_scheduler_type = "cosine",
    warmup_steps = 3,
    per_device_train_batch_size = batch_sizes[0],
    per_device_eval_batch_size = batch_sizes[0],
    num_train_epochs = epoch_num,
    save_strategy = "epoch",
    save_total_limit = 2,
    logging_dir = output_path + "./logs" + postfix,
    logging_steps = 10,
    report_to = "none",              # disable TB in this run
    logging_first_step = True,
    logging_strategy = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model = "accuracy",
    greater_is_better = True,
    gradient_accumulation_steps = 4,
    fp16 = True,
    save_steps = 10,
    dataloader_num_workers = 4,
    seed = random_seed,
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = encoded_train_dataset,
    eval_dataset = encoded_validation_dataset,
    tokenizer = processor,
    compute_metrics = compute_metrics,
    callbacks = [DynamicBatchSizeCallback()],
)

trainer.train()
```

---

## 10) Outputs

- **Best model** (weights & config) saved to:

```python
best_model_path = os.path.join(output_path, "best", postfix)
trainer.save_model(best_model_path)
# → e.g., ensemble/best/_AST_from_SSL_25_02_12-3layers
```

- **Logs** written to `train_log{postfix}.txt` (dynamic batch size + per‑epoch logs).

> If `best_model_path` already exists, it is removed and re‑created to avoid mixing previous runs.

---

## 11) Tips & troubleshooting

- **Check SSL weights**: `model.safetensors` must match the AST encoder you wrap; set `strict=False` when shapes differ at the head.
- **Label drift**: Ensure `id2label` reflects your current folder set; re‑create label maps for each run.
- **OOM errors**: Lower `batch_sizes`, increase `gradient_accumulation_steps`, or reduce `trainNlayers`.
- **Faster debug**: Set `b_just_test=True` to subsample the dataset, and cut `epoch_num`.
- **Metrics choice**: Switch `metric_for_best_model` to `balanced_accuracy` for imbalanced data.
- **Enable TB**: Change `report_to="none"` → `"tensorboard"` to log into `logging_dir`.

---

## 12) Minimal run steps

1. Put SSL checkpoint under `SSL_pretrained_path` (must contain `model.safetensors`).
2. Organize data under `train_dir` / `validation_dir` as in §2.
3. Set `postfix`, `trainNlayers` and freezing options as desired.
4. Run all cells; watch console logs and `train_log{postfix}.txt`.
5. Load the exported best model from `ensemble/best/{postfix}` for downstream evaluation.

