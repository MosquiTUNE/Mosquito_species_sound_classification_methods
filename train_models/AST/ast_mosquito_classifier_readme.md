# AST Mosquito Classifier – README

Train an Audio Spectrogram Transformer (AST) for **multi‑class mosquito species classification** from 1‑second WAV clips.

This notebook/script loads data from a `train/` and a `validation/` folder, preprocesses audio to 16 kHz, fine‑tunes the pretrained model `MIT/ast-finetuned-audioset-10-10-0.4593`, and saves metrics, logs, and checkpoints.

---

## 1) Expected data layout

Your dataset must be organized as class‑labeled subfolders containing `.wav` files:

```
<root>
├── train/
│   ├── species_A/
│   │   ├── a1.wav
│   │   ├── a2.wav
│   │   └── ...
│   └── species_B/
│       ├── b1.wav
│       └── ...
└── validation/
    ├── species_A/
    │   ├── va1.wav
    │   └── ...
    └── species_B/
        ├── vb1.wav
        └── ...
```

---

## 2) Installation

```bash
# Base deps (Torch included via transformers extras)
pip install "transformers[torch]" datasets librosa scikit-learn numpy
```

> **Note:** A CUDA‑enabled PyTorch build is recommended if you have a GPU.

---

## 3) Quick start

1. Set the configurable inputs in the first cell(s):

```python
epoch_num = 40                # Number of training epochs
batch_size = 4                # Per‑device batch size for train/eval
train_dir = "./train-val-independent/train/"
validation_dir = "./train-val-independent/validation/"
everyNth = 1                  # Subsample: use every Nth file (memory saver)
postfix = "-AST_multiclass_mosquito_indep_"  # Suffix for outputs
b_train_just_last = False     # If True: train only the classifier head
trainNlayers = 3              # If >0: unfreeze last N encoder layers; if 0: train full net
```

2. Run the notebook/script. It will:
   - Index training/validation WAVs by class
   - Map class names to ids
   - Load `AutoProcessor` and `AutoModelForAudioClassification`
   - Preprocess each file to 16 kHz and fixed 1‑second tensors
   - Fine‑tune with Hugging Face `Trainer`
   - Evaluate at the end of each epoch

---

## 4) What the inputs control (first cells)

- **`epoch_num`**: Total passes over the training set.
- **`batch_size`**: Per‑device batch size for both training and evaluation.
- **`train_dir` / `validation_dir`**: Paths to your split folders (see layout above).
- **`everyNth`**: Uses only every Nth file when building the datasets (e.g., `everyNth=4` keeps ~25% of files). Helpful if the full set doesn’t fit into RAM or training is too slow.
- **`postfix`**: Appended to output directory/file names to keep runs separate.
- **`b_train_just_last`** (boolean):
  - `True` → freeze everything except the final classification head (fastest, least flexible).
  - `False` → behavior depends on `trainNlayers`.
- **`trainNlayers`** (integer):
  - `> 0` → freeze all layers **except** the last *N* encoder blocks (e.g., `3`).
  - `0` → train the **entire** model end‑to‑end.

> **Mutual behavior:** If `b_train_just_last=True`, it takes precedence and only the head trains. Otherwise, `trainNlayers` decides how much of the encoder is unfrozen.

---

## 5) Preprocessing details

- Files are loaded via **librosa** and resampled to **16 kHz**.
- Each clip is padded or truncated to exactly **1.0 s (16,000 samples)**.
- The AST **processor** converts audio to model inputs; labels are mapped from folder names.

---

## 6) Training configuration (Trainer)

Key defaults you can change in the code:

- Optim/loop via `transformers.Trainer` with:
  - `evaluation_strategy="epoch"`
  - `learning_rate=5e-5`, `lr_scheduler_type="cosine"`, `warmup_steps=3`
  - `gradient_accumulation_steps=8`
  - `fp16=True` (automatic mixed precision if supported)
  - `per_device_train_batch_size=batch_size`
  - `per_device_eval_batch_size=batch_size`
  - `num_train_epochs=epoch_num`
  - `dataloader_num_workers=4`
  - **Metric for model selection:** `balanced_accuracy`

- Metrics logged each eval: `accuracy`, `precision`, `recall`, `f1`, `balanced_accuracy`.

---

## 7) Where outputs are written

The following directories/files are created relative to the working directory:

- **Checkpoints & Trainer state:**
  - `./results{postfix}`  
    Saved **per epoch** (per `save_strategy="epoch"`), with `save_total_limit=2` (keeps the 2 most recent checkpoints).

- **Logs:**
  - `./logs{postfix}`  
    Trainer logs and events. If you enable TensorBoard elsewhere, you can point it here.

- **Final fine‑tuned model & processor:**
  - `./classifier{postfix}`  → `model.save_pretrained(...)`
  - `./classifier-{postfix}` → `processor.save_pretrained(...)`

- **Printed metrics:**
  - After training, `trainer.evaluate()` is called and printed to stdout (accuracy, precision, recall, F1, balanced accuracy).

> Example with the default `postfix = "-AST_multiclass_mosquito_indep_"`:
>
> - Results/checkpoints: `./results-AST_multiclass_mosquito_indep_`
> - Logs: `./logs-AST_multiclass_mosquito_indep_`
> - Final model: `./classifier-AST_multiclass_mosquito_indep_`
> - Final processor: `./classifier--AST_multiclass_mosquito_indep_`  *(note: two dashes in code – adjust if you prefer a single dash)*

---

## 8) GPU usage

The code automatically selects **CUDA** if available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

If you see `Using device: cpu`, ensure your PyTorch build matches your CUDA toolkit and GPU drivers.

---

## 9) Tips & troubleshooting

- **Class mismatch:** Folder names become labels. Ensure train/validation contain the *same* set of class folders.
- **OOM / slow runs:** Increase `everyNth` (e.g., 2, 4, 8), reduce `batch_size`, or train fewer layers (`b_train_just_last=True` or smaller `trainNlayers`).
- **Too little data:** Consider freezing more layers (head‑only or last few blocks) and/or increasing `epoch_num`.
- **Reproducibility:** For strict reproducibility, add manual seeding (`transformers.set_seed`, `numpy.random.seed`, `torch.manual_seed`).
- **Processor/model choice:** This README assumes `MIT/ast-finetuned-audioset-10-10-0.4593`. You can swap to other audio models, but keep shapes/durations consistent.

---

## 10) Citation

If you use this in academic work, please cite ...

---

## 11) Validation notebook — `validate_AST.ipynb`

Use this notebook to **evaluate** a trained checkpoint on a held‑out validation set, compute metrics (overall and class‑wise), visualize confusion matrices, and export per‑file predictions.

### A) Inputs you can set at the top

```python
validation_dir = "../train-val_szunyog_hangok_osztalyozáshoz_3cls_25_02_14/validation/"
checkpoint_dir = "./results_AST_multiclass_mosquito_25_01_28-full-3cls/checkpoint-2092"  # best ckpt
# Optional manual mapping (will be overridden by model.config below):
id2label = {0: 'Aedes_koreicus', 1: 'Ochlerotatus_geniculatus', 2: 'Aedes_albopictus'}
```

- **`validation_dir`**: Folder containing class‑named subfolders with `.wav` files (same layout as training).
- **`checkpoint_dir`**: Path to a **single** HF Trainer checkpoint (e.g., `.../checkpoint-2092`) produced by the training notebook.
- **`id2label` (optional)**: Manual mapping for readability; after loading the checkpoint, the notebook resets to `model.config.id2label` and `model.config.label2id` to ensure consistency with the trained model.
- **`everyNth` (inside loader, default=1)**: If memory or runtime is tight, you may subsample validation by keeping every N‑th file.

> **GPU**: The notebook auto‑selects CUDA if available:
> ```python
> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
> model.to(device)
> ```

### B) Dependencies

`transformers`, `datasets`, `torch`, `librosa`, `numpy`, `pandas`, `scikit-learn`, `tqdm`, `matplotlib`, `seaborn`.

### C) What the notebook does

1. **Loads the checkpoint** (`AutoModelForAudioClassification` and `AutoProcessor`) from `checkpoint_dir`.
2. **Builds the validation dataset** by scanning `validation_dir` and collecting file paths and class labels from folder names.
3. **Runs inference** with a `predict()` helper (and `predict_top()` for Top‑k), using 16 kHz resampling just like training.
4. **Computes metrics**:
   - Overall: `accuracy`, `balanced_accuracy`, `precision` (weighted), `recall` (weighted), `f1` (weighted).
   - **Confusion matrix** (printed and plotted as a heatmap).
   - **Class‑wise**: precision/recall/F1 and a (proxy) ROC‑AUC via one‑vs‑all binarization of *predicted labels* (note: since hard labels are used, ROC‑AUC is illustrative; for true ROC‑AUC use per‑class probabilities/logits).
   - **Top‑k accuracies**: Top‑1/2/3/4.
5. **Per‑file and per‑test grouping**:
   - Saves per‑file predictions to CSV.
   - Extracts a `test_id` from filenames (prefix up to the second underscore), aggregates by `test_id`, and recomputes accuracy/precision/recall/F1 on grouped votes (mode of Top‑1 predictions).
   - Also shows a confusion matrix and class‑wise metrics for grouped results.

### D) Outputs produced

- **Per‑file predictions CSV**:
  - `eval_results_AST.csv` in the working directory with columns:
    - `file` (basename), `true label` (string), `predicted class` (int index), `true class` (int index).
- **Printed metrics** (stdout): overall accuracy, balanced accuracy, precision, recall, F1; Top‑1..Top‑4 accuracies; class‑wise metrics.
- **Plots** (inline):
  - Confusion matrix heatmap for per‑file results (and again for grouped results later on).
  - Line plots of class‑wise metrics (Precision, Recall, F1, ROC‑AUC) per class.

> If you’d like to **persist figures**, add `plt.savefig("<name>.png", dpi=200, bbox_inches="tight")` right before each `plt.show()`.

### E) Expected directory layout for validation

```
validation_dir/
├── Aedes_koreicus/
│   ├── ... .wav
├── Ochlerotatus_geniculatus/
│   ├── ... .wav
└── Aedes_albopictus/
    └── ... .wav
```

Class folder names must match those used during training. The notebook reads **`label2id`/`id2label` from the checkpoint**, so a mismatch between folder names and the checkpoint’s label mapping will produce incorrect metrics.

### F) Notes & tips

- **Label mapping source of truth**: Prefer `model.config.id2label/label2id` from the checkpoint over any manual dicts.
- **Reproducibility**: This notebook evaluates deterministically for a fixed checkpoint and dataset; no seeding is necessary unless you add randomness.
- **Speed**: For quick checks on large sets, increase `everyNth` (e.g., 4 or 8) to subsample.
- **True ROC curves**: For genuine ROC‑AUC per class, collect **probabilities** from `softmax(logits)` and use `roc_auc_score(y_true_onehot, y_score_probs, multi_class="ovr")`.
- **Grouping by `test_id`**: The `test_id` is parsed as the portion of the filename before the **second** underscore; adjust parsing if your naming convention differs.

---

