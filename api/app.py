import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import BeatRequest, PredictionResponse
from api.predictor import predictor

app = FastAPI(
    title="ECG Anomaly Detection API",
    description="CNN-VAE and ResNet-AE based cardiac arrhythmia detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "ECG Anomaly Detection API",
        "models" : ["cnnvae", "resnet"],
        "status" : "running"
    }


@app.get("/health")
def health():
    return {
        "status"      : "healthy",
        "cnnvae_ready": predictor.cnnvae is not None,
        "resnet_ready": predictor.resnet is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: BeatRequest):
    if not request.beats:
        raise HTTPException(status_code=400, detail="No beats provided")

    for beat in request.beats:
        if len(beat) != 180:
            raise HTTPException(
                status_code=400,
                detail=f"Each beat must have 180 samples, got {len(beat)}"
            )

    try:
        result = predictor.predict(request.beats, request.model)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
def list_models():
    return {
        "available_models": [
            {
                "name" : "cnnvae",
                "description": "CNN Variational Autoencoder",
                "auroc" : 0.9036,
                "ready" : predictor.cnnvae is not None
            },
            {
                "name" : "resnet",
                "description": "ResNet Autoencoder",
                "auroc" : 0.8435,
                "ready" : predictor.resnet is not None
            }
        ]
    }