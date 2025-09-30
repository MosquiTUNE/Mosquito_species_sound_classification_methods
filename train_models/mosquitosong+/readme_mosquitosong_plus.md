
# MosquitoSongPlus v2 – README

Train a **custom 1D Convolutional Neural Network (MosquitoSongPlus)** for mosquito species audio classification using CSV file lists generated from a directory of `.wav` clips.

---

## 1) Data structure and CSV generation

The notebook automatically creates two CSV files listing all `.wav` files and their labels:

* `train_fn` → e.g. `./csv_lists_indep/trainData.csv`
* `eval_fn` → e.g. `./csv_lists_indep/valiData.csv`

The generator assumes:

```
train-val-independent/
├── train/
│   ├── Species_A/
│   │   ├── a1.wav
│   │   ├── a2.wav
│   │   └── ...
│   └── Species_B/
│       └── ...
└── validation/
    ├── Species_A/
    │   ├── va1.wav
    └── Species_B/
        └── ...
```

Each subfolder under `train/` and `validation/` is treated as a **class label**. The CSVs produced have the columns:

| Fname              | Genera                        | Species     |
| ------------------ | ----------------------------- | ----------- |
| `path/to/file.wav` | *(left blank in this script)* | `Species_A` |

> The notebook creates the CSVs automatically with:
>
> ```python
> label_list = generate_csv(train_data_dir + "train/", train_fn)
> generate_csv(train_data_dir + "validation/", eval_fn)
> cls_num = len(label_list)
> ```

---

## 2) Key configurable inputs (first cell)

```python
epoch_num = 500         # Number of training epochs
batch_size = 40         # Batch size
learning_rate = 1e-3    # Optimizer learning rate

train_data_dir = "./train-val-independent/"   # Root containing 'train/' and 'validation/' folders

train_fn = "./csv_lists_indep/trainData.csv"  # CSV path for training list
eval_fn = "./csv_lists_indep/valiData.csv"    # CSV path for validation list

str_id = "_msplus_v2_independent"             # Identifier appended to output/log dirs

output_dir = f"./results{str_id}/checkpoints" # Checkpoints saved here
log_dir = f"./results{str_id}/logs"           # TensorBoard logs saved here
```

* **`epoch_num`**: Total training epochs.
* **`batch_size`**: Samples per batch.
* **`learning_rate`**: AdamW learning rate.
* **`train_data_dir`**: Root directory holding `train/` and `validation/`.
* **`train_fn` / `eval_fn`**: File paths for auto-generated CSV lists.
* **`str_id`**: Tag to distinguish different runs (affects `output_dir` and `log_dir`).

---

## 3) Model – `MosquitoSongPlus`

* A **1D CNN** with three convolutional blocks + max-pooling + dropout + 3 fully connected layers.
* Dynamically computes the input size for the first FC layer based on the audio length (`input_size=16000` by default).
* Outputs raw logits for `nn.CrossEntropyLoss`.

Instantiate and move to GPU:

```python
model = MosquitoSongPlus(input_size=16000, num_classes=cls_num).to(device)
```

---

## 4) Data loading & preprocessing

* Uses Hugging Face **`datasets`** to read the generated CSVs.
* Maps `Species` column to integer `label`.
* Loads each audio file at **16 kHz**, normalizes, and pads/truncates to 1 s (16,000 samples).

```python
encoded_train_dataset = train_dataset.map(preprocess_audio_function, ...)
encoded_validation_dataset = validation_dataset.map(preprocess_audio_function, ...)
```

* Custom `collate_fn` stacks waveform arrays into a batch tensor of shape `(batch, 1, 16000)` and produces a label tensor.

---

## 5) Training loop

* Loss: `nn.CrossEntropyLoss()`.

* Optimizer: `torch.optim.AdamW` with `weight_decay=1e-4` and `learning_rate` from above.

* **Metrics computed** each epoch:

  * `accuracy`
  * `precision` (weighted)
  * `recall` (weighted)
  * `f1` (weighted)
  * `balanced_accuracy`

* TensorBoard logging: train/val loss, val accuracy, val balanced accuracy, val F1.

```python
writer = SummaryWriter(log_dir)
```

* **Checkpoints**: The model is saved to `output_dir` whenever validation balanced accuracy improves:

  * `best_model_epoch_X.pt` (best so far)
  * `last_model.pt` (final epoch)

* Console prints:

  * Training loss per epoch
  * Validation loss and metrics
  * Path of new best model

---

## 6) Outputs produced

* **CSV files**:

  * `train_fn` & `eval_fn` lists created automatically from your folder structure.

* **Logs**:

  * TensorBoard event files in `log_dir` → run `tensorboard --logdir ./results_msplus_v2_independent/logs`.

* **Model checkpoints**:

  * Best model at `output_dir/best_model_epoch_X.pt`.
  * Last model at `output_dir/last_model.pt`.

* **Console output**:

  * Per-epoch losses, metrics, and “New best model saved…” notifications.

---

## 7) How to run

1. Place your audio data in `train-val-independent/train/<species>/` and `.../validation/<species>/`.

2. Adjust hyperparameters at the top cell if needed.

3. Run all cells. The script will:

   * Generate CSV lists.
   * Build datasets.
   * Train the MosquitoSongPlus model.
   * Save logs and checkpoints automatically.

4. Monitor training in TensorBoard:

   ```bash
   tensorboard --logdir ./results_msplus_v2_independent/logs
   ```

---

## 8) Tips & extensions

* **Classes**: Make sure the same species folders exist in both train and validation.
* **Audio length**: By default the model expects 1-second 16 kHz clips. Adjust `input_size` if you change it.
* **Learning rate**: 1e-3 is relatively high; reduce if training is unstable.
* **Batch size**: 40 works for moderate GPUs; adjust for your hardware.
* **Early stopping**: Not included, but you can add it by tracking metrics and breaking early.

