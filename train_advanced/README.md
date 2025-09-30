# AST SSL ADVANCED METHODS (SSL) – README

Self‑supervised learning (SSL) pretraining of an **Audio Spectrogram Transformer (AST)** using 1‑second 16 kHz mosquito audio clips. This version uses an independent data split and adds early stopping.

---

## 1) Purpose

- Train AST to reconstruct clean audio spectrogram representations from masked/augmented inputs.
- Learn generalizable representations from unlabeled ABUZZ mosquito sounds.
- Supports **resume from checkpoint** and **early stopping**.

---

## 2) Dataset layout

Put `.wav` files in one folder:

```
./dataset_forSSL_indep/
├── file_0001.wav
├── file_0002.wav
└── ...
```

The script automatically does a **90/10 train/validation split** via `train_test_split`.

---

## 3) Configurable inputs (first cell)

```python
data_dir = "./dataset_forSSL_indep/"   # Directory with all wav files
learning_rate = 1e-4                   # Small LR for stable SSL
batch_size = 8                         # Per-device batch size
epoch_num = 150                        # Number of epochs
str_id = "_SSL_v3_indep"               # ID appended to output dirs
resume_from_checkpoint = None          # Path to resume from or None
```

- **`resume_from_checkpoint`**: e.g. "./AST-SSL-indep_SSL_v3_indep" to reload `model.pth`.
- Seed is fixed at 42 for reproducibility.

---

## 4) Dependencies

- `transformers` (AST model, Trainer, EarlyStoppingCallback)
- `torch`, `torchaudio`
- `librosa`, `numpy`, `scikit-learn`

Ensure GPU + CUDA for speed; FP16 mixed precision is enabled if supported.

---

## 5) Model: `AST_SSL`

A wrapper around the Hugging Face AST encoder with:

- **Encoder**: `ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")`
- **Projector**: linear mapping from encoder hidden size → reconstruction channels.
- **Decoder**: 2‑layer Conv1d over time axis for temporal reconstruction.
- **Loss**: Mean squared error between reconstructed and clean labels.
- **Interpolation**: Aligns decoder output time dimension to labels when needed.

---

## 6) Dataset & augmentation

`AudioDataset` produces:

- **Input**: masked/augmented spectrogram for training.
- **Output**: clean spectrogram as reconstruction target.
- **Loading**: uses `librosa.load(..., sr=16000)`; pads/truncates to 16,000 samples (1 s).
- **Augmentations**: `torchaudio.transforms.TimeMasking` and `FrequencyMasking` (SpecAugment style) applied to input during training.

Custom collator stacks input tensors and labels:

```python
def custom_collate_fn(batch):
    input_values = torch.stack([b["input_values"] for b in batch])
    labels = torch.stack([b["output_values"] for b in batch])
    return {"input_values": input_values, "labels": labels}
```

---

## 7) Training configuration

```python
training_args = TrainingArguments(
    output_dir="./AST-SSL-results" + str_id,
    logging_dir="./AST-SSL-logs" + str_id,
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epoch_num,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="eval_mse",
    greater_is_better=False,
    fp16=True,
    gradient_accumulation_steps=2,
    dataloader_num_workers=4,
    warmup_ratio=0.05,
    max_grad_norm=1.0,
)
```

**Early stopping**:

```python
callbacks = [
    EarlyStoppingCallback(
        early_stopping_patience=10,
        early_stopping_threshold=1e-4
    )
]
```

- **Metric**: `eval_mse` returned by `compute_metrics` (lower is better).
- **Stops** after 10 epochs without significant improvement (threshold 1e‑4).

---

## 8) Running

1. Place `.wav` files in `./dataset_forSSL_indep/`.
2. Adjust parameters in the first cell.
3. Run all cells. Model will train and log metrics.
4. Monitor with TensorBoard:

```bash
tensorboard --logdir ./AST-SSL-logs_SSL_v3_indep
```

---

## 9) Outputs

- **Trainer checkpoints & states**: `./AST-SSL-results{str_id}`
- **TensorBoard logs**: `./AST-SSL-logs{str_id}`
- **Final saved model & processor**: `./AST-SSL-indep{str_id}/`
  - `model.pth` (weights)
  - Hugging Face processor config (`preprocessor_config.json`, etc.)

Example final block:

```python
processor.save_pretrained(model_save_path)
torch.save(ssl_model.state_dict(), os.path.join(model_save_path, "model.pth"))
metrics = trainer.evaluate()
print(metrics)
```

---

## 10) Tips

- Increase `batch_size` if memory allows.
- Freeze encoder layers for faster pretraining.
- Change `target_length` in `AudioDataset` for longer clips.
- Adjust SpecAugment parameters in `AudioDataset` for heavier/no lighter masking.
- Use `resume_from_checkpoint` to continue training from `model.pth`.

---

## 11) Downstream usage

- Fine‑tune a classifier by loading `ssl_model.encoder` weights.
- Perform a linear probe to evaluate representation quality.

---

## 12) Minimal steps to reproduce

```bash
pip install torch torchaudio librosa transformers scikit-learn
python train_ssl_v3_indep.py  # or run notebook cells
```

Check results under `./AST-SSL-indep_SSL_v3_indep/` and logs under `./AST-SSL-logs_SSL_v3_indep/`.

