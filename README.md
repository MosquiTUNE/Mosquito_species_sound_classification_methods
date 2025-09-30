# Mosquito Audio Classification – Code Repository for Paper

This repository contains the **code accompanying a research paper** on mosquito audio classification.  
All training and evaluation scripts used in the experiments are included here.  
The **datasets themselves are hosted elsewhere** ([not included in this repository, but all the data can be downloaded from zenodo](https://zenodo.org/records/17236247?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc1OTI1NDMxMiwiZXhwIjoxOTI0OTA1NTk5fQ.eyJpZCI6IjJmYTA1OTU1LTc0OTUtNGZkZi1iMmM3LTNmY2QyZGQ5YzhhNCIsImRhdGEiOnt9LCJyYW5kb20iOiJhZDAwYjk0Y2IxMWU1ZGYxMTIxMTg2MDUxM2M0MGQzYiJ9.0msG4dqYqwxgpIGimFCKUPgC9VsjTrN1Pf-RnMKEoPyIE6ONymPEy7UBOcqxwzJbLnG7dCo6HOE_Qh12LF1r7A)); some separate README....md like files there describe data used for publication.

---

## Overview of Models & Notebooks

We implemented and evaluated four main model families for mosquito sound classification:

- **MosquitoSong+** – baseline training code
- **ResNet-18** – simple convolutional baseline
- **ResNet-18 + Self-Attention** – ResNet enhanced with attention
- **AST (Audio Spectrogram Transformer)** – SSL-pretraining and fine-tuning

Each model family lives in its own subdirectory under `train_models/` and already has its own README (you can link or open them directly if you want details):

- `train_models/mosquitosong+/readme_mosquitosong_plus.md`
- `train_models/multi-resnet-block/readme_multiresnet.md`
- `train_models/resnet-attention/res_net_self_attention_mosquito_classifier_readme.md`
- `train_models/AST/ast_mosquito_classifier_readme.md`

---

## Advanced Training & Ensemble Setup

Ensemble experiments are managed under `train_advanced/`:

- **SSL pretraining**: `AST_pretrain_with_SSL_v3.ipynb` (produces AST SSL checkpoints)
- **Fine-tuning from SSL**: `ensemble/train_AST_fixed1_from_SSL.ipynb` (single training run, configurable parameters)
- **Papermill multi-run launchers**:  
  - `ensemble/run_train_AST_experiments-independent.ipynb`  
  - `ensemble/run_train_AST_experiments-dependent.ipynb`  
  These scripts call the fine-tuning notebook repeatedly with different parameters to produce multiple fine-tuned models for ensemble evaluation (independent vs. random data split only differ in data paths).
- **Ensemble evaluation**: `ensemble/eval_all_ensemble_AST-fromSSL.ipynb` (aggregates predictions from multiple fine-tuned models and computes metrics)

Older or per-file evaluation scripts are under `train_from_pretrained/`.

---


## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Install torch/torchaudio matching your CUDA:
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchaudio torchvision
```

A unified `requirements.txt` at the project root contains all Python dependencies needed for AST, ResNet, attention, Papermill automation, and evaluation.

---

## How to Reproduce Experiments

1. **Pretrain AST (optional):**  
   Run `train_advanced/AST_pretrain_with_SSL_v3.ipynb` to produce an SSL checkpoint.

2. **Fine-tune models from SSL checkpoint:**  
   Use `ensemble/train_AST_fixed1_from_SSL.ipynb` directly, or launch via Papermill for multiple configs.

3. **Independent vs Random splits:**  
   The two Papermill “runner” notebooks differ only in `train_dir`/`validation_dir` paths.

4. **Evaluate ensemble:**  
   Run `ensemble/eval_all_ensemble_AST-fromSSL.ipynb` pointing to the directory containing your fine-tuned models.

5. **Baselines:**  
   ResNet, ResNet+Attention, and MosquitoSong+ training notebooks live under `train_models/`.

---

## Data

The actual audio datasets are **not included** here. They must be prepared separately in the expected folder structure:

```
<DATA_ROOT>/
├─ train/
│  ├─ Class1/
│  ├─ Class2/
│  └─ ClassN/
└─ validation/
   ├─ Class1/
   ├─ Class2/
   └─ ClassN/
```

Every loader pads or trims to 1-second 16 kHz segments automatically.

---

## Reproducibility Notes

- All AST fine-tuning notebooks set `random_seed=42` and `torch.backends.cudnn.deterministic=True` for reproducibility.
- You can freeze only the classifier, the last N encoder layers, or the whole network depending on your Papermill parameters.
- Papermill generates separate `.ipynb` outputs for each configuration.

---
