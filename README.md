# Mosquito Sound Classification – Code Repository for Paper

This repository contains the **code accompanying a research paper** on mosquito audio classification.  
All training and evaluation scripts used in the experiments are included here.  
The **datasets themselves are hosted elsewhere** ([they are not included in this repository, but all data can be downloaded from Zenodo.
Separate README-style files provided there describe the data used for the publication.](https://zenodo.org/records/17238071?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc1OTI2MTE4OSwiZXhwIjoxOTI0OTA1NTk5fQ.eyJpZCI6Ijc2NTAzNTVjLTcwMDktNGJmZS04N2JhLWMwYmM1OTgwMDU5MCIsImRhdGEiOnt9LCJyYW5kb20iOiIzNTY5ZjY4OGViYTNiMTlmNGI2ZTA1NmUyNmVjNDgyNSJ9.D7Y5dI5GodV6zyVxvjmHx1tMKhJjNcZSPSti2LbufODTFSJJ7C0n3EAriYC1MQmnCxKsdWrhnAJSUYs_3fV1lA))

---

## Overview of Models & Notebooks

We implemented and evaluated four main model families for mosquito sound classification:

- **MosquitoSong+** – training code of publication: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310121
- **ResNet-18** – simple convolutional baseline: https://www.sciencedirect.com/science/article/abs/pii/S1746809424004002
- **ResNet-18 + Self-Attention** – ResNet enhanced with attention: https://github.com/xutong30/WbNet-ResNet-Attention
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

---


Sure—here’s a conda-based install section you can drop into your README:

---

## Installation

Create a **conda** environment and install dependencies:

```bash
# 1) Create and activate the environment
conda create -n mosquito-audio python=3.10 -y
conda activate mosquito-audio

# 2) Install PyTorch stack (choose ONE of the options below)

# (A) CUDA GPU (recommended on NVIDIA GPUs)
#    Pick a CUDA version your driver supports (e.g., 12.1):
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# (B) CPU-only
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch

# (C) Apple Silicon (MPS)
conda install -y pytorch torchvision torchaudio -c pytorch

# 3) Install project Python packages
pip install -r requirements.txt
```

A unified `requirements.txt` at the project root contains almost all Python dependencies needed for **AST**, **ResNet**, **attention modules**, **Papermill automation**, and **evaluation**.
If something extra is required for your setup (e.g., `ffmpeg`), install it with `conda` or `pip`:

```bash
# examples
conda install -y -c conda-forge ffmpeg
pip install <extra-package>
```
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
