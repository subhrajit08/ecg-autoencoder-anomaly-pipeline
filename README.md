# ECG Autoencoder Anomaly Detection Pipeline

Unsupervised cardiac arrhythmia detection using deep learning autoencoders trained on the MIT-BIH Arrhythmia Database. Two architectures compared — CNN-VAE and ResNet-AE — deployed as a production REST API with full MLOps stack.

---

## Overview

- Trained two unsupervised deep learning models on 110,000+ ECG beats from 48 patients
- No anomaly labels used during training — models learn what normal looks like, flag everything else
- Full MLOps pipeline — experiment tracking, containerization, CI/CD, REST API
- Clinical features — heart rate, HRV, NN50, rhythm classification, arrhythmia suggestions

---

## Dataset

[MIT-BIH Arrhythmia Database from PhysioNet](https://physionet.org/content/mitdb/1.0.0/). 48 half-hour ECG recordings, 47 patients, 360 Hz sampling rate.
Beat types classified as normal: N, L, R, e, j. Anomaly: V, A, F, /, S, E, J, a, f.

<img width="1911" height="471" alt="image" src="https://github.com/user-attachments/assets/2ded55f4-f991-4469-9fbe-2d699fe5564d" />


---

## Models

**CNN-VAE** — Convolutional Variational Autoencoder with 4 Conv1d layers, BatchNorm, and probabilistic latent space. Trained exclusively on normal beats. Anomaly score = reconstruction error. Uses Smooth L1 loss and KL divergence.

**ResNet-AE** — Residual Autoencoder with skip connections and double ResBlocks per layer. Deterministic bottleneck, pure reconstruction loss. Richer feature extraction via residual learning.

Both models use min-max normalized anomaly scores and F2-optimized thresholds.



---

## Architecture

```
MIT-BIH Dataset (48 patients, 110,000+ beats)
        ↓
Signal preprocessing
(bandpass filter 0.5-40Hz, R-peak segmentation, z-score normalization)
        ↓
Patient-level train/test split
        ↓
CNN-VAE / ResNet-AE trained on normal beats only
        ↓
Reconstruction error → min-max normalize → threshold → anomaly flag
        ↓
FastAPI REST API → Docker container → GitHub Actions CI/CD
```

---

## Project Structure

```
ecg-autoencoder-anomaly-pipeline/
    src/
        data_loader.py          — download and load MIT-BIH records
        preprocessing.py        — bandpass filter, beat segmentation
        split.py                — patient level train/test split
        vae.py                  — CNN-VAE architecture
        resnet_ae.py            — ResNet-AE architecture
        vae_train.py            — CNN-VAE training loop
        resnet_ae_train.py      — ResNet-AE training loop
        vae_evaluate.py         — evaluation metrics and plots
        resnet_ae_evaluate.py   — evaluation metrics and plots
        explain.py              — HR, HRV, NN50, rhythm classification
    api/
        app.py                  — FastAPI application
        predictor.py            — model inference pipeline
        schemas.py              — request/response models
    models/
        cnnvae_model.pt         — trained CNN-VAE weights
        resnet_ae_model.pt      — trained ResNet-AE weights
    data/splits/
        train_normal.npy        — normal beats for training
        test_beats.npy          — mixed beats for evaluation
        test_labels.npy         — ground truth labels
    outputs/
        vae_evaluation.png
        resnet_evaluation.png
        vae_loss_curve.png
        resnet_loss_curve.png
    Dockerfile
    requirements.txt
    .github/workflows/deploy.yml
```

---

## Installation

```bash
git clone https://github.com/subhrajit08/ecg-autoencoder-anomaly-detection-pipeline.git
cd ecg-autoencoder-anomaly-pipeline
pip install -r requirements.txt
```

---

## Run API locally

```bash
uvicorn api.app:app --reload --port 8000
```

Open `http://localhost:8000/docs`

---

## Run with Docker

```bash
docker build -t ecg-anomaly-detection .
docker run -p 8000:8000 ecg-anomaly-detection
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Model status |
| GET | `/models` | Available models and AUROC |
| POST | `/predict` | Predict from preprocessed beats (180 samples each) |
| POST | `/predict_csv` | Predict from raw ECG CSV file |

**Example — predict from raw ECG CSV:**

```python
import requests

with open('ecg_signal.csv', 'rb') as f:
    response = requests.post(
        "http://localhost:8000/predict_csv",
        files={"file": f},
        params={"model": "cnnvae"}
    )
print(response.json())
```

CSV format — single column of ECG voltage values sampled at 360 Hz:
```
0.123
0.145
0.167
...
```

---

## ECG Analysis Features

Beyond anomaly detection, `explain.py` computes clinical features per recording:

- Heart rate (bpm)
- Average RR interval (ms)
- HRV — SDNN (standard deviation of NN intervals)
- NN50 (successive RR pairs differing more than 50ms)
- R-peak amplitude distribution
- Rhythm classification — Normal Sinus Rhythm, Atrial Fibrillation, Other
- HR condition — Normal, Tachycardia (>100 bpm), Bradycardia (<60 bpm)
- Clinical suggestion based on combined rhythm and HR assessment

---

## Results

| Model | AUROC | AUPRC | Recall | Precision | F2 | Missed Anomalies |
|---|---|---|---|---|---|---|
| CNN-VAE | 0.9036 | 0.5071 | 0.9249 | 0.3553 | 0.7003 | 258 / 3569 |
| ResNet-AE | 0.8435 | 0.3390 | 0.7164 | 0.4281 | 0.6314 | 1012 / 3569 |

CNN-VAE is the primary model — higher AUROC and recall, catches more arrhythmias. ResNet-AE offers higher precision with fewer false alarms. Threshold is tunable based on clinical sensitivity requirements.

---

## MLOps Stack

| Tool | Purpose |
|---|---|
| MLflow | Experiment tracking — loss, AUROC, model artifacts |
| Docker | Containerization — runs identically anywhere |
| GitHub Actions | CI/CD — auto test and build on every push |
| FastAPI | REST API serving both models |

---

## Key Design Decisions

**Unsupervised approach** — models train only on normal beats. No anomaly labels needed during training — realistic for clinical settings where labeled arrhythmia data is scarce.

**Patient-level split** — train and test patients never overlap, preventing data leakage and ensuring honest evaluation.

**Smooth L1 loss** — more robust than MSE for ECG signals with sharp R-peaks. Prevents over-penalization of high-amplitude spikes during training.

**F2 score optimization** — threshold tuned to maximize F2 (recall-weighted) because missing an arrhythmia is clinically more dangerous than a false alarm.

---

## Tech Stack

Python · PyTorch · FastAPI · Docker · MLflow · GitHub Actions · wfdb · scipy · scikit-learn · numpy · matplotlib

---

## References

- Mohebbanaaz, Y. P. S., & Kumari, L. V. R. (2020). **A Review on Arrhythmia Classification Using ECG Signals.** *2020 IEEE International Students' Conference on Electrical, Electronics and Computer Science (SCEECS)*, 1-6. [doi: 10.1109/SCEECS48394.2020.9](https://doi.org/10.1109/SCEECS48394.2020.9)

- Zhou, Z.-H., Wu, J., & Tang, W. (2002). **Ensembling Neural Networks: Many Could Be Better Than All.** *Artificial Intelligence*, 137, 239-263. [doi: 10.1016/S0004-3702(02)00190-X](https://doi.org/10.1016/S0004-3702(02)00190-X)
