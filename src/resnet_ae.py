import torch
import torch.nn as nn
import numpy as np

LATENT_DIM = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResBlock(nn.Module):
    
    def __init__(self, channels, kernel_size=5):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class ResNetEncoder(nn.Module):
    
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.layer1 = nn.Sequential(
            ResBlock(32),
            ResBlock(32),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc_z = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.flatten(x)
        z = self.fc_z(x)
        return z


class ResNetDecoder(nn.Module):
    
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 256 * 11)

        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResBlock(128),
            ResBlock(128),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResBlock(64),
            ResBlock(64),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            ResBlock(32),
            ResBlock(32),

            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 11)
        x = self.deconv(x)
        x = nn.functional.interpolate(x, size=180, mode='linear', align_corners=False)
        x = x.squeeze(1)
        return x


class ResNetAE(nn.Module):

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = ResNetEncoder(latent_dim)
        self.decoder = ResNetDecoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def reconstruction_error(self, x):
        self.eval()
        with torch.no_grad():
            x_recon = self.forward(x)
            error = torch.mean(torch.abs(x - x_recon), dim=1)
        return error

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for name, module in self.encoder.named_modules():
                if any(layer in name for layer in ['stem', 'layer1', 'layer2']):
                    module.eval()


def ae_loss(x, x_recon):
    return nn.functional.smooth_l1_loss(x_recon, x, reduction='sum')


def load_pretrained_encoder(model, pretrained_path):
    
    pretrained = torch.load(pretrained_path, map_location=DEVICE)
    
    if 'model_state' in pretrained:
        pretrained = pretrained['model_state']

    model_dict = model.encoder.state_dict()
    pretrained_dict = {}
    for k, v in pretrained.items():
        if k.startswith('encoder.'):
            stripped_key = k.replace('encoder.', '')
            if stripped_key in model_dict and v.shape == model_dict[stripped_key].shape:
                pretrained_dict[stripped_key] = v

    model.encoder.load_state_dict(pretrained_dict, strict=False)
    print(f"Loaded {len(pretrained_dict)} pretrained layers")

    for name, param in model.encoder.named_parameters():
        if any(layer in name for layer in ['stem', 'layer1', 'layer2']):
            param.requires_grad = False

    frozen = sum(1 for p in model.encoder.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.encoder.parameters() if p.requires_grad)
    print(f"Frozen params: {frozen}")
    print(f"Trainable params: {trainable}")

    return model


if __name__ == "__main__":

    model = ResNetAE(latent_dim=LATENT_DIM).to(DEVICE)
    print(model)

    dummy = torch.randn(8, 180).to(DEVICE)
    x_recon = model(dummy)

    print(f"\nInput shape: {dummy.shape}")
    print(f"Reconstructed: {x_recon.shape}")

    loss = ae_loss(dummy, x_recon)
    print(f"Loss: {loss.item():.4f}")

    scores = model.reconstruction_error(dummy)
    print(f"Anomaly scores: {scores.shape}")
    print(f"Score values: {scores}")
