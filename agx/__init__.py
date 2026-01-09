"""
AG-X 2026: Advanced Gravity Research Simulation Platform
=========================================================

A production-grade Python platform for theoretical exploration, simulation,
analysis, and AI-assisted optimization of gravity-alteration and hypothetical
anti-gravity systems.

⚠️ DISCLAIMER: All simulations and results are THEORETICAL and EDUCATIONAL only.
No claims of real-world anti-gravity creation or violation of known physical laws.

Modules:
--------
- physics: Multi-scale gravity modeling (Newtonian, GR, Quantum, Speculative)
- ai: Deep learning exploration, RL optimization, anomaly detection, XAI
- viz: 2D/3D/4D visualization, interactive dashboards
- experiments: Reproducible experiment management, reports generation
- advanced: Multi-agent simulation, evolutionary algorithms, symbolic math

Usage:
------
    from agx import PhysicsEngine, AIOptimizer, Visualizer
    
    # Create a simulation
    engine = PhysicsEngine(config="configs/default.yaml")
    results = engine.run_simulation(timesteps=1000)
    
    # Visualize results
    viz = Visualizer()
    viz.render_spacetime(results)

Author: AG-X Research Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "AG-X Research Team"
__license__ = "MIT"

# Scientific validity disclaimer - displayed on import
DISCLAIMER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  AG-X 2026 - THEORETICAL SIMULATION PLATFORM                                 ║
║                                                                              ║
║  ⚠️  All results are SPECULATIVE and for EDUCATIONAL purposes only.          ║
║  This platform does NOT claim to create real anti-gravity effects.           ║
║  All physics models include hypothetical constructs clearly labeled.         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Core imports (lazy loading for performance)
def __getattr__(name: str):
    """Lazy loading of submodules for faster import times."""
    if name == "PhysicsEngine":
        from agx.physics import PhysicsEngine
        return PhysicsEngine
    elif name == "AIOptimizer":
        from agx.ai import AIOptimizer
        return AIOptimizer
    elif name == "Visualizer":
        from agx.viz import Visualizer
        return Visualizer
    elif name == "ExperimentManager":
        from agx.experiments import ExperimentManager
        return ExperimentManager
    elif name == "config":
        from agx import config
        return config
    raise AttributeError(f"module 'agx' has no attribute '{name}'")

__all__ = [
    "__version__",
    "__author__",
    "DISCLAIMER",
    "PhysicsEngine",
    "AIOptimizer", 
    "Visualizer",
    "ExperimentManager",
]
