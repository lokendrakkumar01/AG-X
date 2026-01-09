"""
Explainable AI Module
======================

Provides interpretable explanations for AI decisions and simulation results.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field


@dataclass
class Explanation:
    """Structured explanation of AI/simulation results."""
    summary: str
    feature_importance: Dict[str, float]
    key_factors: List[str]
    confidence: float
    detailed_explanation: str
    visualizations: Dict[str, Any] = field(default_factory=dict)


class AIExplainer:
    """
    Explainable AI for Physics Simulations.
    
    Provides SHAP-like feature importance, sensitivity analysis,
    and natural language explanations.
    """
    
    def __init__(self, method: str = "permutation"):
        self.method = method
        self.feature_names: List[str] = []
    
    def set_feature_names(self, names: List[str]) -> None:
        self.feature_names = names
    
    def explain_prediction(self, 
                          model_fn: Callable[[np.ndarray], np.ndarray],
                          X: np.ndarray,
                          baseline: Optional[np.ndarray] = None,
                          n_samples: int = 100) -> Explanation:
        """
        Explain a model prediction using permutation importance.
        
        Args:
            model_fn: Model prediction function
            X: Input data point(s)
            baseline: Baseline values for comparison
            n_samples: Number of permutation samples
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_features = X.shape[1]
        if baseline is None:
            baseline = np.zeros(n_features)
        
        # Original prediction
        original_pred = model_fn(X).mean()
        
        # Permutation importance
        importances = {}
        for i in range(n_features):
            X_permuted = X.copy()
            errors = []
            
            for _ in range(n_samples // n_features):
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                permuted_pred = model_fn(X_permuted).mean()
                errors.append(abs(original_pred - permuted_pred))
            
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            importances[feature_name] = np.mean(errors)
        
        # Normalize importances
        total = sum(importances.values()) or 1
        importances = {k: v/total for k, v in importances.items()}
        
        # Sort by importance
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        key_factors = [f"{k} ({v:.1%})" for k, v in sorted_features[:5]]
        
        # Generate summary
        top_feature = sorted_features[0][0] if sorted_features else "unknown"
        summary = f"The prediction is most influenced by {top_feature}."
        
        # Detailed explanation
        detailed = self._generate_detailed_explanation(sorted_features, original_pred)
        
        return Explanation(
            summary=summary,
            feature_importance=importances,
            key_factors=key_factors,
            confidence=0.8,  # Confidence in explanation
            detailed_explanation=detailed,
        )
    
    def _generate_detailed_explanation(self, 
                                       sorted_features: List[tuple],
                                       prediction: float) -> str:
        """Generate detailed natural language explanation."""
        lines = [
            "## Explanation Summary\n",
            f"**Predicted Value:** {prediction:.4f}\n",
            "\n### Key Contributing Factors:\n",
        ]
        
        for i, (feature, importance) in enumerate(sorted_features[:5], 1):
            lines.append(f"{i}. **{feature}**: {importance:.1%} contribution\n")
        
        lines.extend([
            "\n### Interpretation:\n",
            "The model's prediction is primarily driven by the factors listed above. ",
            "Higher importance values indicate features that, when changed, ",
            "would have a larger impact on the prediction.\n",
            "\n⚠️ *Note: These explanations are approximate and for educational purposes.*"
        ])
        
        return "".join(lines)
    
    def sensitivity_analysis(self, 
                            model_fn: Callable,
                            center_point: np.ndarray,
                            perturbation_range: float = 0.1,
                            n_points: int = 20) -> Dict[str, np.ndarray]:
        """
        Perform sensitivity analysis around a center point.
        
        Returns how the output changes as each input is varied.
        """
        n_features = len(center_point)
        results = {}
        
        for i in range(n_features):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            
            # Create perturbation range
            min_val = center_point[i] * (1 - perturbation_range)
            max_val = center_point[i] * (1 + perturbation_range)
            perturbations = np.linspace(min_val, max_val, n_points)
            
            outputs = []
            for val in perturbations:
                perturbed = center_point.copy()
                perturbed[i] = val
                output = model_fn(perturbed.reshape(1, -1))
                outputs.append(output.mean())
            
            results[feature_name] = {
                "inputs": perturbations,
                "outputs": np.array(outputs),
                "gradient": np.gradient(outputs, perturbations).mean(),
            }
        
        return results
    
    def explain_simulation_state(self, state: Dict[str, Any]) -> str:
        """Generate natural language explanation of simulation state."""
        lines = ["## Simulation State Analysis\n"]
        
        if "energy" in state:
            energy = state["energy"]
            lines.append(f"**Total Energy:** {energy:.4e} units\n")
            if energy < 0:
                lines.append("*The system is bound (negative total energy).*\n")
        
        if "bodies" in state:
            n_bodies = len(state["bodies"])
            lines.append(f"**Number of Bodies:** {n_bodies}\n")
        
        if "chaos_analysis" in state:
            chaos = state["chaos_analysis"]
            if chaos.get("is_chaotic"):
                lines.append("⚠️ **Warning:** Chaotic behavior detected!\n")
                lines.append(f"Lyapunov exponent: {chaos.get('lyapunov_exponent', 'N/A')}\n")
        
        if "theoretical_note" in state or "disclaimer" in state:
            lines.append("\n---\n*All results are theoretical simulations only.*")
        
        return "".join(lines)
    
    def explain(self, model_fn: Callable, X: np.ndarray, **kwargs) -> Explanation:
        """Convenience method for explain_prediction."""
        return self.explain_prediction(model_fn, X, **kwargs)
