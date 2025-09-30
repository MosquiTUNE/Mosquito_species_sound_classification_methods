
# Multi-class Mosquito Audio Classifier – MultiResNet (Paper Version)

This notebook trains a **multi-class mosquito species classifier** based on a multi-branch / ResNet-style architecture. It is intended to reproduce or extend the results described in the accompanying paper.

---

## 1) Expected data layout

Your dataset should be organized into `train/` and `validation/` subfolders, with each species as a separate subfolder containing `.wav` clips:

```
data_root/
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

The notebook will scan these folders and build the training/validation sets automatically.

---

## 2) Configurable inputs (first cells)

Typical hyperparameters and paths defined at the top:

```python
# Number of training epochs
epoch_num =  ...

# Batch size
batch_size =  ...

# Learning rate
learning_rate = ...

# Root directory containing 'train' and 'validation'
data_dir = "./train-val-independent/"

# Optional CSV list files if used
train_csv = "./csv_lists_indep/trainData.csv"
val_csv   = "./csv_lists_indep/valiData.csv"

# Identifier for this run (used to name output folders)
str_id = "_multiresnet_paper"

# Output and log directories
output_dir = f"./results{str_id}/checkpoints"
log_dir    = f"./results{str_id}/logs"
```

Adjust these before running:

* **`epoch_num`**: total epochs to train.
* **`batch_size`**: samples per batch.
* **`learning_rate`**: AdamW optimizer learning rate.
* **`data_dir`**: root folder of your data.
* **`train_csv` / `val_csv`**: optional file lists (script can generate them if not present).
* **`str_id`**: run tag appended to output and log directories.

---

## 3) Model architecture

* A **MultiResNet** classifier (paper variant), implemented in PyTorch.
* Accepts 1-second 16 kHz audio clips (or configurable).
* Combines multiple convolutional/ResNet blocks and fully connected layers to output logits for *N* classes.
* Number of classes is inferred from your dataset.

---

## 4) Data loading and preprocessing

* Automatically creates CSV lists of all files under `train/` and `validation/` if not already existing.
* Maps each species (folder name) to an integer label.
* Loads each `.wav` at **16 kHz**, normalizes, and pads/truncates to 1 second.
* Uses a custom `collate_fn` to produce `(batch, 1, 16000)` float tensors and integer labels.

---

## 5) Training loop and metrics

* Loss: `nn.CrossEntropyLoss`.
* Optimizer: `torch.optim.AdamW` with `weight_decay`.
* Tracks metrics each epoch:

  * Accuracy
  * Precision (weighted)
  * Recall (weighted)
  * F1 (weighted)
  * Balanced accuracy
* Logs all metrics to TensorBoard at `log_dir`.

Checkpoint saving:

* Best model saved when validation balanced accuracy improves:

  * `output_dir/best_model_epoch_X.pt`
* Last model always saved as:

  * `output_dir/last_model.pt`

Console prints:

* Training and validation loss per epoch.
* Metrics per epoch.
* “New best model saved” message when balanced accuracy improves.

---

## 6) Outputs produced

* **CSV lists**: for train and validation (if generated).
* **Checkpoints**: best and last model `.pt` files in `output_dir`.
* **TensorBoard logs**: metrics and losses in `log_dir`.
* **Console metrics**: printed each epoch.

You can visualize training progress with:

```bash
tensorboard --logdir ./results_multiresnet_paper/logs
```

---

## 7) How to run

1. Place your data under `train/` and `validation/` subfolders as described.
2. Adjust the hyperparameters and paths in the first cell.
3. Run the notebook sequentially:

   * It will generate CSV lists (if needed).
   * Build the datasets.
   * Instantiate the MultiResNet model.
   * Train and evaluate each epoch.
   * Save logs and checkpoints automatically.

---

## 8) Tips & notes

* **Classes**: Make sure all classes present in training are also in validation.
* **Audio length**: Default 1-second clips at 16 kHz. Change `input_size` if using different lengths.
* **Learning rate**: Lower if training becomes unstable.
* **Batch size**: Adjust based on GPU memory.
* **Early stopping**: Not included; add your own if needed.
* **Validation script**: For evaluation only, use a notebook similar to `validate_AST.ipynb`, pointing it to the best checkpoint.

