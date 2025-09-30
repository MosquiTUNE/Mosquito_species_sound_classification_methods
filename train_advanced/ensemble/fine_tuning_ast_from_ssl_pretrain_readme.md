# Unified README — AST Ensemble Training & Evaluation

This README explains how the provided notebooks implement **ensemble training** and **evaluation** using the Audio Spectrogram Transformer (AST). The workflow begins from an **SSL‑pretrained model** and uses Papermill to parameterize runs, enabling multiple fine‑tuned models for ensemble evaluation.

---

## Included Notebooks

- **`train_AST_fixed1_from_SSL.ipynb`** — the core training notebook. It fine‑tunes an AST classifier starting from a self‑supervised (SSL) pretrained encoder. All key hyperparameters can be overridden externally (via Papermill) to create multiple variants for ensemble learning.
- **`run_train_AST_experiments-independent.ipynb`** — Papermill wrapper script to launch ~10 training runs on the **independent split** dataset. Identical code to the dependent version except for the dataset path.
- **`run_train_AST_experiments-dependent.ipynb`** — Papermill wrapper script to launch ~10 training runs on the **random/dependent split** dataset.
- **`eval_all_ensemble_AST-fromSSL.ipynb`** — loads all fine‑tuned models produced by the training notebooks and computes ensemble predictions/metrics.

---

## 1. Training from SSL Pretrained Model

`train_AST_fixed1_from_SSL.ipynb`:
- **Loads** an SSL‑pretrained AST model (projector/decoder removed, encoder retained).
- **Adds** a linear classification head with configurable number of classes.
- **Freezing strategy**: configurable to train only the classifier, the last *N* layers, or the full encoder.
- **Parameters** (set in the first cell):
  - `SSL_pretrained_path`: checkpoint folder of the pretrained model (expects `model.safetensors`).
  - `train_dir` / `validation_dir`: folder‑per‑class datasets.
  - `postfix`: run ID suffix for outputs.
  - `b_train_just_last`: boolean to train only the classifier.
  - `trainNlayers`: integer to train last *N* encoder layers.
  - `epoch_num`, `batch_sizes`: epoch count and per‑epoch batch size schedule.
  - `random_seed`, `output_path`.
- **Papermill usage**: Pass different parameter sets (paths, seeds, layers to fine‑tune, etc.) to launch multiple distinct training runs for ensemble creation.

The notebook uses Hugging Face’s `Trainer` and a custom audio preprocessing function to map file paths into tensors. Metrics (accuracy, precision, recall, F1, balanced accuracy) are automatically computed per epoch. The best model is saved under `output_path/best/<postfix>`.

---

## 2. Papermill Experiment Launchers

- **Independent split launcher** (`run_train_AST_experiments-independent.ipynb`) and **dependent split launcher** (`run_train_AST_experiments-dependent.ipynb`) are thin wrappers.
- They simply **loop through 10 different parameter combinations** (learning rate, layers to train, seeds, etc.) and **execute `train_AST_fixed1_from_SSL.ipynb` via Papermill**.
- The only difference between the two is the **dataset path**:
  - Independent split ensures no data leakage between train/validation.
  - Dependent (random) split allows within‑recording mixing.
- The output of each run is a trained model folder named after its `postfix`, ready for ensemble evaluation.

---

## 3. Ensemble Evaluation

`eval_all_ensemble_AST-fromSSL.ipynb`:
- **Loads** all saved fine‑tuned models from a specified directory.
- **Runs** predictions on the validation set.
- **Aggregates** outputs to produce ensemble predictions (e.g. majority vote or averaged probabilities).
- **Computes** metrics: Accuracy, Balanced Accuracy, weighted Precision/Recall/F1, Confusion Matrix, per‑class metrics (Precision, Recall, F1, ROC‑AUC).
- **Visualizes** results with confusion matrices and metric curves per class.

This notebook lets you quickly benchmark the ensemble of models trained with different hyperparameters on either the independent or dependent split.

---

## 4. Typical Workflow

1. **Pretrain** (done once): produce an SSL AST model.
2. **Fine‑tune multiple models**: run `run_train_AST_experiments-independent.ipynb` and/or `run_train_AST_experiments-dependent.ipynb` with Papermill to generate ~10 variants each.
3. **Evaluate ensemble**: run `eval_all_ensemble_AST-fromSSL.ipynb` to load all variants and compute ensemble metrics.

---

## 5. Environment & Requirements

- Python ≥ 3.9
- `torch`, `torchaudio`, `librosa`
- `transformers`, `datasets`, `safetensors`, `scikit‑learn`, `matplotlib`, `seaborn`, `tqdm`
- Papermill for parameterized execution.

Example install:
```bash
pip install torch torchaudio librosa transformers datasets safetensors scikit-learn matplotlib seaborn tqdm papermill
```

---

## 6. Outputs

- **Fine‑tuned models** in `output_path` with distinct `postfix` names.
- **Logs** of dynamic batch size and training metrics.
- **Evaluation plots**: confusion matrix and class‑wise metrics.
- **Ensemble performance metrics** across all models.

---

This setup allows easy experimentation with AST fine‑tuning strategies and immediate ensemble evaluation on mosquito sound classification datasets.

