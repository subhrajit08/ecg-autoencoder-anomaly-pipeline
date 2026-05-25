import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
from vae import CNNVAE, vae_loss, LATENT_DIM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 200
BATCH_SIZE = 512
LEARNING_RATE = 0.001
PATIENCE = 10
MODELS_DIR = './models'
SPLITS_DIR = './data/splits'


def load_training_data():
    
    train = np.load(f'{SPLITS_DIR}/train_normal.npy')
    tensor = torch.FloatTensor(train)
    dataset = TensorDataset(tensor)
    loader  = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=2,
            )
    
    print(f"Training beats: {train.shape[0]:,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batches per epoch: {len(loader)}")
    
    return loader


def training_step(model, loader, optimizer, epoch):
    
    model.train()
    total_loss = 0

    for batch_idx, (x,) in enumerate(loader):
        x = x.to(DEVICE)
        
        x_recon, mu, logvar = model(x)
        loss = vae_loss(x, x_recon, mu, logvar, beta=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


def training():

    os.makedirs(MODELS_DIR, exist_ok=True)

    loader = load_training_data()

    model = CNNVAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 45)
    
    mlflow.set_experiment("ecg-anomaly-detection")

    with mlflow.start_run(run_name="cnnvae"):
        mlflow.log_param("model", "CNN-VAE")
        mlflow.log_param("latent_dim", LATENT_DIM)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("patience", PATIENCE)

        best_loss = float('inf')
        patience_counter = 0
        loss_history = []

        for epoch in range(1, EPOCHS + 1):

            avg_loss = training_step(model, loader, optimizer, epoch)
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
                }, f'{MODELS_DIR}/cnnvae_model.pt')
                print("Saved Successfully!")
                patience_counter = 0
            
            else:
                patience_counter += 1
                print(f"(No Improvement {patience_counter}/{PATIENCE})")
            
            print()
            
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

        mlflow.log_metric("best_loss", best_loss)
        mlflow.pytorch.log_model(model, "cnnvae_model")
        mlflow.log_artifact(f'{MODELS_DIR}/cnnvae_model.pt')

    print("=" * 45)
    print(f"Training complete. Best loss: {best_loss:.6f}")
    print(f"Model saved to {MODELS_DIR}/cnnvae_model.pt")
    
    plot_loss_curve(loss_history)

def plot_loss_curve(loss_history):
    
    os.makedirs('./outputs', exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color='#00d4aa', linewidth=1)
    plt.title('Training Loss Curve', color='white')
    plt.xlabel('Epoch', color='gray')
    plt.ylabel('Loss', color='gray')
    plt.tick_params(colors='gray')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('./outputs/vae_loss_curve.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f0f')
    print("Loss curve saved to ./outputs/vae_loss_curve.png")


if __name__ == "__main__":
    
    print(f"Using device: {DEVICE}")
    training()
