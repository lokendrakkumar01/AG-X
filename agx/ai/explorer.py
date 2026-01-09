"""
Parameter Space Explorer with Deep Learning
============================================

Neural network surrogate models and Bayesian optimization
for efficient parameter space exploration.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ExplorationResult:
    """Result from parameter space exploration."""
    best_parameters: np.ndarray
    best_value: float
    all_samples: np.ndarray
    all_values: np.ndarray
    uncertainty: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class SurrogateModel(nn.Module):
    """Neural network surrogate model for fast function approximation."""
    
    def __init__(self, input_dim: int, output_dim: int = 1, 
                 hidden_layers: List[int] = [256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EnsembleModel:
    """Ensemble of surrogate models for uncertainty estimation."""
    
    def __init__(self, input_dim: int, n_models: int = 5, **kwargs):
        self.models = [SurrogateModel(input_dim, **kwargs) for _ in range(n_models)]
        self.n_models = n_models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for model in self.models:
            model.to(self.device)
    
    def train_all(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
                  batch_size: int = 32, lr: float = 1e-3):
        """Train all ensemble models."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for model in self.models:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            model.train()
            for _ in range(epochs):
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    pred = model(batch_X)
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    optimizer.step()
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimation."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy()
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean = predictions.mean(axis=0).squeeze()
        std = predictions.std(axis=0).squeeze()
        
        return mean, std


class ParameterExplorer:
    """
    Parameter Space Explorer with Bayesian Optimization.
    
    Uses neural network ensemble as surrogate model and
    expected improvement for acquisition.
    """
    
    def __init__(self, input_dim: int = 10, n_ensemble: int = 5):
        self.input_dim = input_dim
        self.ensemble: Optional[EnsembleModel] = None
        self.n_ensemble = n_ensemble
        
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []
        self.best_value = float('-inf')
        self.best_params: Optional[np.ndarray] = None
    
    def set_bounds(self, bounds: np.ndarray):
        """Set parameter bounds: (n_params, 2) array of [min, max]."""
        self.bounds = bounds
        self.input_dim = len(bounds)
    
    def _sample_random(self, n_samples: int) -> np.ndarray:
        """Sample random points within bounds."""
        samples = np.random.uniform(
            self.bounds[:, 0], 
            self.bounds[:, 1],
            size=(n_samples, self.input_dim)
        )
        return samples
    
    def _expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Calculate expected improvement acquisition function."""
        if self.ensemble is None or len(self.y_observed) == 0:
            return np.ones(len(X))
        
        mean, std = self.ensemble.predict(X)
        std = np.maximum(std, 1e-9)
        
        improvement = mean - self.best_value - xi
        Z = improvement / std
        
        from scipy.stats import norm
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        
        return ei
    
    def explore(self, objective_fn: Callable[[np.ndarray], float],
                bounds: np.ndarray,
                n_initial: int = 20,
                n_iterations: int = 50,
                batch_size: int = 5) -> ExplorationResult:
        """
        Explore parameter space using Bayesian optimization.
        
        Args:
            objective_fn: Function to maximize, takes params returns scalar
            bounds: Parameter bounds (n_params, 2)
            n_initial: Number of initial random samples
            n_iterations: Number of BO iterations
            batch_size: Samples per iteration
        """
        self.set_bounds(bounds)
        
        # Initial random sampling
        X_init = self._sample_random(n_initial)
        y_init = np.array([objective_fn(x) for x in X_init])
        
        self.X_observed.extend(X_init)
        self.y_observed.extend(y_init)
        
        best_idx = np.argmax(y_init)
        self.best_value = y_init[best_idx]
        self.best_params = X_init[best_idx]
        
        # Initialize ensemble
        self.ensemble = EnsembleModel(self.input_dim, self.n_ensemble)
        
        # Bayesian optimization loop
        for iteration in range(n_iterations):
            # Train surrogate
            X_train = np.array(self.X_observed)
            y_train = np.array(self.y_observed)
            self.ensemble.train_all(X_train, y_train, epochs=50)
            
            # Sample candidates and compute acquisition
            candidates = self._sample_random(100)
            ei_values = self._expected_improvement(candidates)
            
            # Select top candidates
            top_indices = np.argsort(ei_values)[-batch_size:]
            X_new = candidates[top_indices]
            
            # Evaluate objective
            y_new = np.array([objective_fn(x) for x in X_new])
            
            self.X_observed.extend(X_new)
            self.y_observed.extend(y_new)
            
            # Update best
            if y_new.max() > self.best_value:
                best_idx = np.argmax(y_new)
                self.best_value = y_new[best_idx]
                self.best_params = X_new[best_idx]
        
        # Final uncertainty estimation
        X_all = np.array(self.X_observed)
        y_all = np.array(self.y_observed)
        _, uncertainty = self.ensemble.predict(X_all)
        
        return ExplorationResult(
            best_parameters=self.best_params,
            best_value=self.best_value,
            all_samples=X_all,
            all_values=y_all,
            uncertainty=uncertainty,
            metadata={
                "n_evaluations": len(self.y_observed),
                "n_iterations": n_iterations,
            }
        )
    
    def sensitivity_analysis(self, center: np.ndarray, 
                            perturbation: float = 0.1) -> Dict[str, float]:
        """Analyze parameter sensitivity around a point."""
        if self.ensemble is None:
            return {}
        
        sensitivities = {}
        base_val, _ = self.ensemble.predict(center.reshape(1, -1))
        
        for i in range(self.input_dim):
            perturbed = center.copy()
            perturbed[i] *= (1 + perturbation)
            new_val, _ = self.ensemble.predict(perturbed.reshape(1, -1))
            
            sensitivities[f"param_{i}"] = abs(new_val - base_val) / perturbation
        
        return sensitivities
