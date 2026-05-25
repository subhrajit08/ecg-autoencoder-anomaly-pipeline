from pydantic import BaseModel
from typing import List

class BeatRequest(BaseModel):
    beats: List[List[float]]
    model: str = "cnnvae"

class BeatPrediction(BaseModel):
    beat_index: int
    anomaly_score: float
    is_anomaly: bool

class PredictionResponse(BaseModel):
    model_used: str
    total_beats: int
    anomalies_detected: int
    anomaly_rate: float
    threshold: float
    predictions: List[BeatPrediction]