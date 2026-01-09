"""
Anomaly Detection Module
=========================

Detects unusual behaviors in simulation data using autoencoders and statistical methods.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class AnomalyResult:
    """Result from anomaly detection."""
    anomaly_scores: np.ndarray
    is_anomaly: np.ndarray
    threshold: float
    anomaly_indices: np.ndarray
    reconstruction_errors: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Autoencoder(nn.Module):
    """Autoencoder for anomaly detection via reconstruction error."""
    
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        reconstructed, _ = self.forward(x)
        return torch.mean((x - reconstructed) ** 2, dim=1)


class AnomalyDetector:
    """
    Anomaly Detection for Simulation Data.
    
    Uses autoencoder reconstruction error and statistical thresholds
    to identify unusual simulation states.
    """
    
    def __init__(self, method: str = "autoencoder", contamination: float = 0.01):
        self.method = method
        self.contamination = contamination
        self.model: Optional[Autoencoder] = None
        self.threshold = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, epochs: int = 100, lr: float = 1e-3) -> None:
        """Train anomaly detector on normal data."""
        input_dim = X.shape[1]
        self.model = Autoencoder(input_dim).to(self.device)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            reconstructed, _ = self.model(X_tensor)
            loss = criterion(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()
        
        # Set threshold based on training data
        self.model.eval()
        with torch.no_grad():
            errors = self.model.get_reconstruction_error(X_tensor).cpu().numpy()
        
        # Threshold at (1-contamination) percentile
        self.threshold = np.percentile(errors, (1 - self.contamination) * 100)
        self._is_fitted = True
    
    def detect(self, X: np.ndarray) -> AnomalyResult:
        """Detect anomalies in new data."""
        if not self._is_fitted:
            # Fit on the data if not already fitted
            self.fit(X)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            errors = self.model.get_reconstruction_error(X_tensor).cpu().numpy()
        
        is_anomaly = errors > self.threshold
        anomaly_indices = np.where(is_anomaly)[0]
        
        # Normalize scores to 0-1 range
        max_error = max(errors.max(), self.threshold * 2)
        scores = errors / max_error
        
        return AnomalyResult(
            anomaly_scores=scores,
            is_anomaly=is_anomaly,
            threshold=self.threshold,
            anomaly_indices=anomaly_indices,
            reconstruction_errors=errors,
            metadata={
                "n_anomalies": len(anomaly_indices),
                "contamination": self.contamination,
            }
        )
    
    def detect_in_timeseries(self, X: np.ndarray, window_size: int = 10) -> AnomalyResult:
        """Detect anomalies in time series with sliding window."""
        n_samples = len(X)
        features = []
        
        for i in range(n_samples - window_size + 1):
            window = X[i:i+window_size]
            feat = np.concatenate([
                window.mean(axis=0),
                window.std(axis=0),
                window[-1] - window[0],  # Trend
            ])
            features.append(feat)
        
        features = np.array(features)
        result = self.detect(features)
        
        # Map back to original indices
        original_indices = result.anomaly_indices + window_size - 1
        
        return AnomalyResult(
            anomaly_scores=result.anomaly_scores,
            is_anomaly=result.is_anomaly,
            threshold=result.threshold,
            anomaly_indices=original_indices,
            reconstruction_errors=result.reconstruction_errors,
            metadata={"window_size": window_size, **result.metadata}
        )
