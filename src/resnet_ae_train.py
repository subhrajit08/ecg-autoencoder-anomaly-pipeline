import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
from resnet_ae import ResNetAE, ae_loss, LATENT_DIM, DEVICE, load_pretrained_encoder

EPOCHS = 200
BATCH_SIZE = 512
LEARNING_RATE = 0.005
PATIENCE = 10
MODELS_DIR = './models'
SPLITS_DIR = './data/splits'
PRETRAINED = None


def load_training_data():
    train = np.load(f'{SPLITS_DIR}/train_normal.npy')
    tensor = torch.FloatTensor(train)
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )
    print(f"Training beats: {train.shape[0]:,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batches per epoch: {len(loader)}")
    return loader


def training_step(model, loader, optimizer):
    model.train()
    total_loss = 0
    noise_factor = 0.4
    for (x,) in loader:
        x = x.to(DEVICE)
        x_noisy = x + noise_factor * torch.randn_like(x)
        x_recon = model(x_noisy) 
        loss = ae_loss(x, x_recon)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader.dataset)


def training():
    os.makedirs(MODELS_DIR, exist_ok=True)
    loader = load_training_data()
    model = ResNetAE(latent_dim=LATENT_DIM).to(DEVICE)

    if PRETRAINED and os.path.exists(PRETRAINED):
        model = load_pretrained_encoder(model, PRETRAINED)
        print("Transfer learning — early layers frozen")
    else:
        print("No pretrained weights — training from scratch")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    print(f"Epochs: {EPOCHS}")
    print(f"LR: {LEARNING_RATE}")
    print("=" * 45)
    
    mlflow.set_experiment("ecg-anomaly-detection")

    with mlflow.start_run(run_name="resnet"):
        mlflow.log_param("model", "ResNet-AE")
        mlflow.log_param("latent_dim", LATENT_DIM)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("patience", PATIENCE)

        best_loss = float('inf')
        patience_counter = 0
        loss_history = []

        for epoch in range(1, EPOCHS + 1):
            avg_loss = training_step(model, loader, optimizer)
            loss_history.append(avg_loss)
            scheduler.step(avg_loss)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            print(f"Epoch {epoch:>3}/{EPOCHS} | Loss: {avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    'best_loss': best_loss,
                }, f'{MODELS_DIR}/resnet_ae_model.pt')
                print("Saved Successfully!")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"(No Improvement {patience_counter}/{PATIENCE})")

            print()

            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break
    
        mlflow.log_metric("best_loss", best_loss)
        mlflow.pytorch.log_model(model, "resnet_ae_model")
        mlflow.log_artifact(f'{MODELS_DIR}/resnet_ae_model.pt')

    print("=" * 45)
    print(f"Training complete. Best loss: {best_loss:.6f}")
    plot_loss_curve(loss_history)


def plot_loss_curve(loss_history):
    os.makedirs('./outputs', exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color='#00d4aa', linewidth=1)
    plt.title('ResNet-AE Training Loss', color='white')
    plt.xlabel('Epoch', color='gray')
    plt.ylabel('Loss', color='gray')
    plt.tick_params(colors='gray')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('./outputs/resnet_loss_curve.png', dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
    print("Loss curve saved to ./outputs/resnet_loss_curve.png")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    training()