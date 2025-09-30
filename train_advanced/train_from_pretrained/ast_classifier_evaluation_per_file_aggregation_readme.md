# AST Classifier Evaluation & Per‑File Aggregation — README

Evaluate a fine‑tuned **Audio Spectrogram Transformer (AST)** classifier on a validation set, compute overall and class‑wise metrics, visualize a confusion matrix, and produce **per‑file** predictions with optional **test‑id grouping** (majority vote).

---

## 1) What this script does

- Loads a trained AST‑based classifier checkpoint (weights in **Safetensors** format).
- Builds a Hugging Face `Dataset` from a folder tree of `.wav` files.
- Runs inference to compute:
  - Overall metrics: Accuracy, Balanced Accuracy, Precision/Recall/F1 (weighted)
  - Confusion matrix heatmap
  - Class‑wise Precision/Recall/F1 and a proxy ROC‑AUC (via one‑vs‑all binarized labels)
- **Per‑file** Top‑k predictions and **grouped evaluation** by a `test_id` parsed from filenames (prefix up to the second underscore).

---

## 2) Expected validation layout

```
validation_dir/
├── Aedes_albopictus/
│   ├── ... .wav
├── Aedes_koreicus/
│   ├── ... .wav
└── Ochlerotatus_geniculatus/
    └── ... .wav
```

Each subfolder name is treated as the **label**. Ensure labels match those used during training.

---

## 3) Key configurable inputs

```python
validation_dir = "../../dataset/train-val_mosquito_sounds_classification_3cls_25_02_14/validation/"
checkpoint_dir = "./ensemble/best/_independentdata_seed42_adamw_cosine/"  # contains model.safetensors

id2label = {
    0: 'Aedes_albopictus',
    1: 'Aedes_koreicus',
    2: 'Ochlerotatus_geniculatus'
}
label2id = {v: k for k, v in id2label.items()}
```

- **`validation_dir`**: root folder of validation audio split.
- **`checkpoint_dir`**: folder containing `model.safetensors`.
- **`id2label` / `label2id`**: mapping between class indices and folder names.

---

## 4) Model assembly (AST encoder → classifier)

1. Load the AST encoder and **discard** projector/decoder pieces from SSL wrapper.
2. Attach a fresh linear **classifier head** sized to `num_classes`.
3. Load weights from `model.safetensors` with `strict=False` (to allow head shape differences).

Skeleton:

```python
processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
base_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

class AST_SSL(nn.Module):
    ...  # encoder + (removed) projector/decoder

ssl_model = AST_SSL(base_model, output_dim=[768])
del ssl_model.projector; del ssl_model.decoder

class AST_Classifier(nn.Module):
    def __init__(self, ssl_model, num_classes):
        super().__init__()
        self.encoder = ssl_model.encoder
        self.classifier = nn.Linear(768, num_classes)
    def forward(self, input_values, labels=None):
        x = self.encoder(input_values).last_hidden_state
        x = self.encoder.layernorm(x)
        x = x.mean(dim=1)  # temporal average pooling
        logits = self.classifier(x)
        return {"logits": logits} if labels is None else {"loss": F.cross_entropy(logits, labels), "logits": logits}

model = AST_Classifier(ssl_model, num_classes=len(id2label))
state_dict = load_file(checkpoint_dir + "model.safetensors")
model.load_state_dict(state_dict, strict=False)
model.to(device)
```

---

## 5) Build the validation dataset

```python
def load_audio_dataset_from_folders(validation_dir, everyNth=1):
    def scan(dir_):
        rows = []
        for cls in os.listdir(dir_):
            cls_path = os.path.join(dir_, cls)
            if os.path.isdir(cls_path):
                for fn in os.listdir(cls_path):
                    if fn.endswith('.wav'):
                        rows.append({"file_path": os.path.join(cls_path, fn), "label": cls})
        return rows
    data = scan(validation_dir)
    return Dataset.from_dict({
        "file_path": [d["file_path"] for i, d in enumerate(data) if i % everyNth == 0],
        "label":     [d["label"]     for i, d in enumerate(data) if i % everyNth == 0],
    })

validation_dataset = load_audio_dataset_from_folders(validation_dir)
```

---

## 6) Inference helper

```python
def predict(filepath, model, processor, device):
    wav, _ = librosa.load(filepath, sr=16000)
    inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_values)["logits"]
    probs = torch.softmax(logits, dim=-1)
    cls_idx = int(torch.argmax(probs, dim=-1).item())
    cls_prob = float(probs[0, cls_idx].item())
    return cls_idx, cls_prob
```

---

## 7) Metrics & plots

After collecting `true_labels` and `predicted_labels` (indices), compute:

- **Accuracy**, **Balanced Accuracy**, **Precision/Recall/F1 (weighted)**
- **Confusion matrix** (Seaborn heatmap)
- **Class‑wise** Precision/Recall/F1 and a proxy **ROC‑AUC** using one‑vs‑all binarization of hard labels

> Note: For a proper ROC‑AUC, aggregate **probabilities** per class across samples instead of hard labels.

---

## 8) Per‑file Top‑k and grouped evaluation

**Top‑k predictions per file**:

```python
def predict_top(filepath, model, processor, device):
    wav, _ = librosa.load(filepath, sr=16000)
    inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_values)["logits"]
    probs = torch.softmax(logits, dim=-1).cpu().numpy().ravel()
    order = np.argsort(probs)[::-1]
    return [int(i) for i in order], [float(probs[i]) for i in order]
```

**Aggregate by `test_id`** (portion of filename before the **second underscore**):

```python
data = []
for ex in validation_dataset:
    fp = ex["file_path"]; true_lbl = label2id[ex["label"]]
    top_idx, top_probs = predict_top(fp, model, processor, device)
    base = os.path.basename(fp); parts = base.split("_")
    test_id = parts[0] + "_" + parts[1] if len(parts) > 2 else "N/A"
    data.append({
        "file_path": fp,
        "test_id": test_id,
        "true_label": true_lbl,
        "top1_prediction": top_idx[0],
        "top2_prediction": top_idx[1] if len(top_idx) > 1 else None,
        "top3_prediction": top_idx[2] if len(top_idx) > 2 else None,
        "top4_prediction": top_idx[3] if len(top_idx) > 3 else None,
    })

df_results = pd.DataFrame(data)

# Majority vote per test_id
grouped_results = df_results.groupby("test_id").agg(
    true_label=("true_label", lambda x: x.iloc[0]),
    predicted_label=("top1_prediction", lambda x: x.mode()[0] if not x.mode().empty else "N/A"),
).reset_index()

# Overall grouped metrics
accuracy = (grouped_results.true_label == grouped_results.predicted_label).mean()
precision = grouped_results.groupby("predicted_label").apply(lambda g: (g.true_label == g.predicted_label).mean()).mean()
recall = grouped_results.groupby("true_label").apply(lambda g: (g.true_label == g.predicted_label).mean()).mean()
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
print({"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1})
```

You can then re‑compute per‑class metrics and plot another confusion matrix based on the **grouped** predictions.

---

## 9) Outputs

- **Printed metrics** for per‑file and/or grouped evaluations
- **Confusion matrix figure(s)** shown inline
- **`df_results`** (optional: save to CSV) with per‑file Top‑k predictions

```python
df_results.to_csv("eval_per_file_topk.csv", index=False)
```

---

## 10) Tips

- Ensure `id2label` matches the training label order.
- If labels differ across splits, remap folder names before evaluation.
- For true ROC‑AUC, collect **per‑class probabilities** and call `roc_auc_score(y_true_onehot, y_score_probs, multi_class="ovr")`.
- Adjust the `test_id` parsing rule to your filename convention.

---

## 11) Minimal run steps

1. Set `validation_dir`, `checkpoint_dir`, and `id2label`.
2. Run the dataset loader, build the model, and load `model.safetensors`.
3. Execute the inference loop to collect predictions.
4. Compute metrics and visualize.
5. (Optional) Run per‑file Top‑k and grouped aggregation, and export CSV.

