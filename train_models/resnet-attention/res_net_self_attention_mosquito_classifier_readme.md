# ResNet + Self‑Attention Mosquito Classifier — README

Train a multi‑class **mosquito species** classifier using a **ResNet‑18 backbone** with an optional **self‑attention** head. The notebook/script builds CSV lists from your folder structure, loads data via a custom dataset, trains the model, and writes detailed logs plus checkpoints.

---

## 1) Directory layout & CSVs

Expected input layout (per‑class folders of `.wav` files):

```
./train-val-independent/
├── train/
│   ├── Species_A/
│   │   ├── a1.wav
│   │   └── ...
│   └── Species_B/
│       └── ...
└── validation/
    ├── Species_A/
    │   ├── va1.wav
    └── Species_B/
        └── ...
```

The script **auto‑generates** CSVs listing all files and their labels (columns: `Fname, Genera, Species`):

- Train list → `./csv_resnet_indep/trainData.csv`
- Validation list → `./csv_resnet_indep/valiData.csv`

> The class/label map (`label_list`) is inferred from the **train** folder names and is reused for validation.

---

## 2) Configurable inputs (first cells)

```python
train_prefix = "multiclass_indep_classifier_"  # prefixes output files/folders

train_folder = "./train-val-independent/train/"
valid_folder = "./train-val-independent/validation/"
train_csv   = "./csv_resnet_indep/trainData.csv"
valid_csv   = "./csv_resnet_indep/valiData.csv"

b_selfattention = True        # enable self-attention head in the model

batch_size = 4
learning_rate = 1e-3
epochs = 500

start_model_path = None  # path to a pre-trained binary model (.pth) to fine-tune from
```

- **`train_prefix`**: Used to name log files and checkpoint directory.
- **`train_folder` / `valid_folder`**: Where class subfolders live.
- **`train_csv` / `valid_csv`**: Generated file lists the script will use.
- **`b_selfattention`**: Toggles the attention module inside the ResNet (`model_modified.resnet18_attention`).
- **`batch_size`**, **`learning_rate`**, **`epochs`**: Standard training hyperparameters.
- **`start_model_path`**: If set, loads a **binary** (2‑class) checkpoint, then replaces the final linear layer for multi‑class training (weights re‑initialized).

---

## 3) Dependencies / modules

Built‑ins and common libs:

- `torch`, `torchvision` (via your custom ResNet), `numpy`, `scikit‑learn`, `tqdm`

Project‑local modules (expected in the working directory / PYTHONPATH):

- `dataPreprocess` → provides `ListDataset(train_csv, label_map, split)` that loads waveforms and labels
- `model_modified` → provides `resnet18_attention(in_channels, num_classes, b_selfattention)`

> Ensure `dataPreprocess.py` and `model_modified.py` are available alongside the notebook/script.

---

## 4) Data prep flow

1. **CSV generation** (`generate_csv`) scans each class folder and writes rows like:
   - `Fname=<full/path.wav>`, `Genera=""` (unused), `Species=<folder_name>`
2. The **label map** is `{class_name: idx}` from the **train** set.
3. Dataloaders are created via `dataPreprocess.ListDataset(...)` and `DataLoader(..., batch_size, shuffle)`.

> Validation uses the same label mapping as training; make sure class folders match between splits.

---

## 5) Model & initialization

- Default: `resnet_model = model_modified.resnet18_attention(1, cls_num, b_selfattention=b_selfattention)`
- **Warm‑start from binary model** (if `start_model_path` is provided):
  1. Load `resnet18_attention(1, 2, ...)` weights.
  2. Replace classification layer to `nn.Linear(..., out_features=cls_num)` and **Xavier init** weights.

The model runs on GPU if available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = resnet_model.to(device)
```

---

## 6) Training loop & scheduler

- Loss: `nn.CrossEntropyLoss()`
- Optimizer: `torch.optim.Adam(lr=1e-3)`
- LR scheduler: `CosineAnnealingLR(optimizer, T_max=200, eta_min=0)`
- Per‑epoch reporting (printed and logged):
  - **Train**: loss, accuracy
  - **Validation**: loss, accuracy, classification report, confusion matrix, epoch time

**Metrics source:**
- Classification report: `sklearn.metrics.classification_report`
- Confusion matrix: `sklearn.metrics.confusion_matrix`

---

## 7) Outputs & where they go

**Folders**
- Checkpoints: `./{train_prefix}checkpoints/`
  - `best_loss_{epoch}.pth` → best **validation loss** so far
  - `big_{epoch}.pth` → best **validation accuracy** so far
  - `last_model.pth` → always overwritten with the latest epoch

**Logs**
- Running log: `./{train_prefix}output_big.txt`
  - Per‑epoch train/val loss & accuracy
  - Per‑epoch classification report & confusion matrix (validation)
  - Per‑epoch time usage
- Final summary: `./{train_prefix}output.txt` (created at the very end)
  - Best validation accuracy
  - Classification report for the best‑acc model
  - A closing note about the saved best model

> **Note:** The end‑of‑training note mentions `resnet_attention.pth`; the actual best‑accuracy files are named `big_{epoch}.pth`.

---

## 8) How to run

1. Place your data under `train/` and `validation/` as shown above.
2. Adjust the inputs in Section 2 (paths, hyperparameters). Optionally set `start_model_path`.
3. Run all cells. The script will:
   - Generate CSVs
   - Build datasets and dataloaders
   - Instantiate the model (with/without warm‑start)
   - Train for `epochs`
   - Save logs and checkpoints continuously

---

## 9) Tips & troubleshooting

- **Class coverage:** Ensure identical class sets in training and validation.
- **Batch size & LR:** With long clips or many classes, lower `batch_size` and/or `learning_rate` if training is unstable.
- **Scheduler horizon:** `T_max=200` (for cosine annealing) is independent of `epochs`; you can tune it to your run length.
- **Best‑loss guard:** `best_loss_val` is set on **epoch 1**; the conditional uses `epoch==1 or valid_loss < best_loss_val` to avoid referencing before assignment.
- **Custom dataset:** Any audio normalization/resampling is handled in `dataPreprocess.ListDataset`; confirm it matches your training assumptions (e.g., 16 kHz, 1 s).
- **Reproducibility:** Add manual seeding if exact repeatability is needed.

---

## 10) After training

- Use your existing validation/evaluation notebook (e.g., a variant of `validate_AST.ipynb`) by pointing it to the chosen checkpoint from `./{train_prefix}checkpoints/`.
- For per‑file vs grouped evaluations (e.g., by test id), reuse the grouping logic from your AST validator.

