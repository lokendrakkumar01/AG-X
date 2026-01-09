"""
AG-X 2026 Visualization Module
===============================

2D/3D/4D visualization for particles, spacetime, fields, and simulations.
"""

from agx.viz.renderer import Renderer, RenderConfig
from agx.viz.dashboard import Dashboard, create_app
from agx.viz.temporal import TemporalVisualizer, AnimationConfig

__all__ = [
    "Renderer",
    "RenderConfig", 
    "Dashboard",
    "create_app",
    "TemporalVisualizer",
    "AnimationConfig",
]

# Convenience class
class Visualizer:
    """Unified visualization interface."""
    
    def __init__(self):
        self.renderer = Renderer()
        self.temporal = TemporalVisualizer()
    
    def render_particles(self, *args, **kwargs):
        return self.renderer.render_particles(*args, **kwargs)
    
    def render_spacetime(self, *args, **kwargs):
        return self.renderer.render_spacetime_grid(*args, **kwargs)
    
    def render_field(self, *args, **kwargs):
        return self.renderer.render_scalar_field(*args, **kwargs)
    
    def create_animation(self, *args, **kwargs):
        return self.temporal.create_animation(*args, **kwargs)
