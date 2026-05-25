import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from vae import CNNVAE, LATENT_DIM as VAE_LATENT_DIM
from resnet_ae import ResNetAE, LATENT_DIM as RESNET_LATENT_DIM
from scipy.signal import butter, filtfilt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
THRESHOLD_PERCENTILE = 80

FS     = 360
WINDOW = 90

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=FS):
    nyq  = fs / 2.0
    b, a = butter(4, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)

def segment_signal(signal):
    from scipy.signal import find_peaks
    sig = signal[:, 0]
    filtered = bandpass_filter(signal)
    sig_f = filtered[:, 0]
    r_peaks, _ = find_peaks(sig_f, distance=int(FS*0.5), height=np.mean(sig_f))
    beats = []
    for peak in r_peaks:
        start, end = peak - WINDOW, peak + WINDOW
        if start >= 0 and end <= len(sig_f):
            beat = sig_f[start:end]
            mean = beat.mean()
            std  = beat.std() + 1e-8
            beats.append((beat - mean) / std)
    return np.array(beats, dtype=np.float32)

class ModelPredictor:

    def __init__(self):
        self.cnnvae = None
        self.resnet = None
        self.load_models()

    def load_models(self):
        cnnvae_path = os.path.join(MODELS_DIR, 'cnnvae_model.pt')
        resnet_path = os.path.join(MODELS_DIR, 'resnet_ae_model.pt')

        if os.path.exists(cnnvae_path):
            self.cnnvae = CNNVAE(latent_dim=VAE_LATENT_DIM).to(DEVICE)
            checkpoint = torch.load(cnnvae_path, map_location=DEVICE)
            self.cnnvae.load_state_dict(checkpoint['model_state'])
            self.cnnvae.eval()
            print(f"CNN-VAE loaded")

        if os.path.exists(resnet_path):
            self.resnet = ResNetAE(latent_dim=RESNET_LATENT_DIM).to(DEVICE)
            checkpoint = torch.load(resnet_path, map_location=DEVICE)
            self.resnet.load_state_dict(checkpoint['model_state'])
            self.resnet.eval()
            print(f"ResNet-AE loaded")

    def get_errors(self, model, beats_array):
        errors = []
        batch_size = 256
        for i in range(0, len(beats_array), batch_size):
            batch = torch.FloatTensor(beats_array[i:i+batch_size]).to(DEVICE)
            error = model.reconstruction_error(batch)
            errors.append(error.cpu().numpy())
        errors = np.concatenate(errors)
        errors = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)
        return errors

    def predict(self, beats: list, model_name: str):
        beats_array = np.array(beats, dtype=np.float32)

        if model_name == "cnnvae" and self.cnnvae:
            model = self.cnnvae
        elif model_name == "resnet" and self.resnet:
            model = self.resnet
        else:
            raise ValueError(f"Model {model_name} not available")

        errors = self.get_errors(model, beats_array)
        threshold = float(np.percentile(errors, THRESHOLD_PERCENTILE))
        predictions = []

        for i, score in enumerate(errors):
            predictions.append({
                'beat_index'   : i,
                'anomaly_score': round(float(score), 4),
                'is_anomaly'   : bool(score >= threshold)
            })

        anomalies = sum(1 for p in predictions if p['is_anomaly'])

        return {
            'model_used'        : model_name,
            'total_beats'       : len(beats),
            'anomalies_detected': anomalies,
            'anomaly_rate'      : round(anomalies / len(beats), 4),
            'threshold'         : round(threshold, 4),
            'predictions'       : predictions
        }
    
    def predict_from_signal(self, signal, model_name):
        beats_array = segment_signal(signal)

        if len(beats_array) == 0:
            raise ValueError("No beats detected in signal")

        return self.predict(beats_array.tolist(), model_name)


predictor = ModelPredictor()