import torch
import torch.nn as nn

LATENT_DIM = 16

class Encoder(nn.Module):
    
    def __init__(self, latent_dim = LATENT_DIM):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        self.flatten_size = 256 * 12
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.05) 
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
    
    def forward(self, x):
        
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dropout(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Decoder(nn.Module):
    
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        
        self.flatten_size = 256 * 12
        
        self.fc = nn.Linear(latent_dim, self.flatten_size)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1,  kernel_size=5, stride=2, padding=2, output_padding=1),
        )
    
    def forward(self, z):
        
        x = self.fc(z)
        x = x.view(x.size(0), 256, 12)
        x = self.deconv(x)
        x = x[:, :, :180]
        x = x.squeeze(1)
        
        return x


class CNNVAE(nn.Module):
    
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def reparameterize(self, mu, logvar):
        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z
        
        else:
            return mu 
    
    def forward(self, x):
        
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar
    
    def reconstruction_error(self, x, alpha=0.1):
        
        self.eval()
        
        with torch.no_grad():
            x_recon, mu, logvar = self.forward(x)
            recon_error = torch.mean(torch.abs(x - x_recon), dim=1)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        return recon_error + (alpha * kl_div)


def vae_loss(x, x_recon, mu, logvar, beta=0.2):
    
    recon_loss = nn.functional.smooth_l1_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + (beta * kl_loss)


if __name__ == "__main__":
    
    model = CNNVAE(latent_dim=LATENT_DIM)
    print(model)
    
    dummy = torch.randn(8, 180)
    x_recon, mu, logvar = model(dummy)

    print(f"\nInput shape: {dummy.shape}")
    print(f"Reconstructed: {x_recon.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")

    loss = vae_loss(dummy, x_recon, mu, logvar)
    print(f"Loss: {loss.item():.4f}")

    scores = model.reconstruction_error(dummy)
    print(f"Anomaly scores: {scores.shape}")
    print(f"Score values: {scores}")
