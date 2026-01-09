"""
AG-X 2026 AI Intelligence Layer
================================

Deep learning exploration, reinforcement learning optimization,
anomaly detection, and explainable AI insights.
"""

from agx.ai.explorer import ParameterExplorer, SurrogateModel
from agx.ai.optimizer import RLOptimizer, OptimizationResult
from agx.ai.anomaly import AnomalyDetector, AnomalyResult
from agx.ai.explainer import AIExplainer, Explanation

__all__ = [
    "ParameterExplorer",
    "SurrogateModel",
    "RLOptimizer", 
    "OptimizationResult",
    "AnomalyDetector",
    "AnomalyResult",
    "AIExplainer",
    "Explanation",
]

# Convenience class combining all AI capabilities
class AIOptimizer:
    """
    Unified AI Optimizer combining exploration, optimization, and explanation.
    """
    
    def __init__(self):
        self.explorer = ParameterExplorer()
        self.optimizer = RLOptimizer()
        self.detector = AnomalyDetector()
        self.explainer = AIExplainer()
    
    def explore_parameters(self, *args, **kwargs):
        return self.explorer.explore(*args, **kwargs)
    
    def optimize(self, *args, **kwargs):
        return self.optimizer.optimize(*args, **kwargs)
    
    def detect_anomalies(self, *args, **kwargs):
        return self.detector.detect(*args, **kwargs)
    
    def explain(self, *args, **kwargs):
        return self.explainer.explain(*args, **kwargs)
