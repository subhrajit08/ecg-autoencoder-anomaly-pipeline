import mlflow
import mlflow.pytorch
import torch
import sys
import os
sys.path.append('./src')

from vae import CNNVAE, LATENT_DIM as VAE_LATENT_DIM
from resnet_ae import ResNetAE, LATENT_DIM as RESNET_LATENT_DIM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mlflow.set_experiment("ecg-anomaly-detection")

with mlflow.start_run(run_name="cnnvae"):
    mlflow.log_param("model", "CNN-VAE")
    mlflow.log_param("latent_dim", VAE_LATENT_DIM)
    mlflow.log_param("batch_size", 512)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_metric("auroc", 0.9036)
    mlflow.log_metric("auprc", 0.5071)
    mlflow.log_metric("f2", 0.7003)
    mlflow.log_metric("precision", 0.3553)
    mlflow.log_metric("recall", 0.9249)
    mlflow.log_artifact("./outputs/vae_evaluation.png")
    mlflow.log_artifact("./outputs/vae_loss_curve.png")
    checkpoint = torch.load('./models/cnnvae_model.pt', map_location=DEVICE)
    mlflow.log_metric("best_loss", checkpoint['best_loss'])
    mlflow.log_metric("best_epoch", checkpoint['epoch'])
    model = CNNVAE(latent_dim=VAE_LATENT_DIM).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    mlflow.pytorch.log_model(model, "cnnvae_model")
    print("CNN-VAE logged")

with mlflow.start_run(run_name="resnet_ae"):
    mlflow.log_param("model", "ResNet-AE")
    mlflow.log_param("latent_dim", RESNET_LATENT_DIM)
    mlflow.log_param("batch_size", 512)
    mlflow.log_param("learning_rate", 0.005)
    mlflow.log_metric("auroc", 0.8435)
    mlflow.log_metric("auprc", 0.3390)
    mlflow.log_metric("f2", 0.6314)
    mlflow.log_metric("precision", 0.4281)
    mlflow.log_metric("recall", 0.7164)
    mlflow.log_artifact("./outputs/resnet_evaluation.png")
    mlflow.log_artifact("./outputs/resnet_loss_curve.png")
    checkpoint = torch.load('./models/resnet_ae_model.pt', map_location=DEVICE)
    mlflow.log_metric("best_loss", checkpoint['best_loss'])
    mlflow.log_metric("best_epoch", checkpoint['epoch'])
    model = ResNetAE(latent_dim=RESNET_LATENT_DIM).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    mlflow.pytorch.log_model(model, "resnet_ae_model")
    print("ResNet-AE logged")

print("All experiments logged successfully")