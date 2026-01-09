"""
Core Visualization Renderer
============================

Renders particles, spacetime grids, vector fields, and energy heatmaps.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class RenderConfig:
    """Visualization configuration."""
    theme: str = "dark"
    colormap: str = "viridis"
    particle_size: float = 8.0
    vector_scale: float = 0.3
    grid_resolution: int = 32
    width: int = 900
    height: int = 700
    show_axes: bool = True
    show_grid: bool = True


class Renderer:
    """
    Core Visualization Renderer for AG-X simulations.
    
    Supports 2D and 3D rendering of particles, fields, and spacetime.
    """
    
    DARK_TEMPLATE = {
        "paper_bgcolor": "#1a1a2e",
        "plot_bgcolor": "#16213e",
        "font_color": "#e0e0e0",
    }
    
    LIGHT_TEMPLATE = {
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#f5f5f5",
        "font_color": "#333333",
    }
    
    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        self.template = self.DARK_TEMPLATE if self.config.theme == "dark" else self.LIGHT_TEMPLATE
    
    def _apply_theme(self, fig: go.Figure) -> go.Figure:
        """Apply theme to figure."""
        fig.update_layout(
            paper_bgcolor=self.template["paper_bgcolor"],
            plot_bgcolor=self.template["plot_bgcolor"],
            font=dict(color=self.template["font_color"]),
            width=self.config.width,
            height=self.config.height,
        )
        return fig
    
    def render_particles_2d(self, 
                           positions: np.ndarray,
                           masses: Optional[np.ndarray] = None,
                           velocities: Optional[np.ndarray] = None,
                           labels: Optional[List[str]] = None) -> go.Figure:
        """Render particles in 2D."""
        n = len(positions)
        
        if masses is None:
            masses = np.ones(n)
        
        # Color by mass (including negative for exotic matter)
        colors = masses
        colorscale = [[0, "blue"], [0.5, "white"], [1, "red"]]
        
        # Size by absolute mass
        sizes = np.abs(masses) / np.abs(masses).max() * self.config.particle_size + 5
        
        fig = go.Figure()
        
        # Particles
        fig.add_trace(go.Scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                colorscale=colorscale,
                colorbar=dict(title="Mass"),
                line=dict(width=1, color="white"),
            ),
            text=labels or [f"Body {i}" for i in range(n)],
            hovertemplate="<b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>",
        ))
        
        # Velocity vectors
        if velocities is not None:
            for i in range(n):
                fig.add_annotation(
                    x=positions[i, 0],
                    y=positions[i, 1],
                    ax=positions[i, 0] + velocities[i, 0] * self.config.vector_scale,
                    ay=positions[i, 1] + velocities[i, 1] * self.config.vector_scale,
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowcolor="#00ff88",
                )
        
        fig.update_layout(
            title="Particle Positions (2D)",
            xaxis_title="X",
            yaxis_title="Y",
            showlegend=False,
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        return self._apply_theme(fig)
    
    def render_particles_3d(self,
                           positions: np.ndarray,
                           masses: Optional[np.ndarray] = None,
                           velocities: Optional[np.ndarray] = None,
                           trails: Optional[np.ndarray] = None) -> go.Figure:
        """Render particles in 3D with optional trails."""
        n = len(positions)
        
        if masses is None:
            masses = np.ones(n)
        
        sizes = np.abs(masses) / np.abs(masses).max() * self.config.particle_size + 3
        
        fig = go.Figure()
        
        # Particle trails
        if trails is not None:
            for i in range(min(n, len(trails))):
                trail = trails[i]  # Shape: (timesteps, 3)
                fig.add_trace(go.Scatter3d(
                    x=trail[:, 0], y=trail[:, 1], z=trail[:, 2],
                    mode="lines",
                    line=dict(width=2, color="rgba(100, 200, 255, 0.5)"),
                    showlegend=False,
                ))
        
        # Particles
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            marker=dict(
                size=sizes,
                color=masses,
                colorscale="Plasma",
                colorbar=dict(title="Mass"),
                line=dict(width=1, color="white"),
            ),
        ))
        
        fig.update_layout(
            title="Particle Positions (3D)",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube",
            ),
        )
        
        return self._apply_theme(fig)
    
    def render_spacetime_grid(self,
                             X: np.ndarray,
                             Y: np.ndarray,
                             curvature: np.ndarray,
                             as_surface: bool = True) -> go.Figure:
        """Render spacetime curvature grid."""
        fig = go.Figure()
        
        if as_surface:
            # 3D surface where height represents curvature
            Z = -curvature * 5  # Invert and scale for "dip" visualization
            
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                colorscale="Viridis",
                colorbar=dict(title="Curvature"),
                opacity=0.9,
            ))
            
            fig.update_layout(
                title="Spacetime Curvature (Embedding Diagram)",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Curvature Depth",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                ),
            )
        else:
            # 2D heatmap
            fig.add_trace(go.Heatmap(
                x=X[0], y=Y[:, 0], z=curvature,
                colorscale="Viridis",
                colorbar=dict(title="Curvature"),
            ))
            
            fig.update_layout(
                title="Spacetime Curvature (Heatmap)",
                xaxis_title="X",
                yaxis_title="Y",
            )
            fig.update_yaxes(scaleanchor="x")
        
        return self._apply_theme(fig)
    
    def render_scalar_field(self,
                           X: np.ndarray,
                           Y: np.ndarray,
                           field: np.ndarray,
                           title: str = "Scalar Field") -> go.Figure:
        """Render a 2D scalar field as heatmap."""
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            x=X[0], y=Y[:, 0], z=field,
            colorscale=self.config.colormap,
            colorbar=dict(title="Value"),
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="X",
            yaxis_title="Y",
        )
        fig.update_yaxes(scaleanchor="x")
        
        return self._apply_theme(fig)
    
    def render_vector_field(self,
                           X: np.ndarray,
                           Y: np.ndarray,
                           U: np.ndarray,
                           V: np.ndarray,
                           title: str = "Vector Field") -> go.Figure:
        """Render a 2D vector field with quiver-like arrows."""
        fig = go.Figure()
        
        # Subsample for clarity
        step = max(1, len(X) // 20)
        x = X[::step, ::step].flatten()
        y = Y[::step, ::step].flatten()
        u = U[::step, ::step].flatten()
        v = V[::step, ::step].flatten()
        
        # Magnitude for coloring
        magnitude = np.sqrt(u**2 + v**2)
        
        # Create arrows using annotations
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            marker=dict(size=3, color=magnitude, colorscale="Hot"),
        ))
        
        scale = self.config.vector_scale
        for i in range(len(x)):
            if magnitude[i] > 1e-10:
                fig.add_annotation(
                    x=x[i], y=y[i],
                    ax=x[i] + u[i] * scale,
                    ay=y[i] + v[i] * scale,
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowcolor="rgba(255, 100, 100, 0.7)",
                )
        
        fig.update_layout(title=title, xaxis_title="X", yaxis_title="Y")
        fig.update_yaxes(scaleanchor="x")
        
        return self._apply_theme(fig)
    
    def render_energy_history(self,
                             times: np.ndarray,
                             energies: Dict[str, np.ndarray]) -> go.Figure:
        """Render energy components over time."""
        fig = go.Figure()
        
        colors = {"kinetic": "#ff6b6b", "potential": "#4ecdc4", "total": "#ffe66d"}
        
        for name, values in energies.items():
            fig.add_trace(go.Scatter(
                x=times,
                y=values,
                mode="lines",
                name=name.capitalize(),
                line=dict(color=colors.get(name, "#ffffff"), width=2),
            ))
        
        fig.update_layout(
            title="Energy Evolution",
            xaxis_title="Time",
            yaxis_title="Energy",
            legend=dict(x=0.02, y=0.98),
        )
        
        return self._apply_theme(fig)
    
    def render_particles(self, positions: np.ndarray, **kwargs) -> go.Figure:
        """Convenience method - auto-selects 2D or 3D."""
        if positions.shape[1] >= 3 and np.any(positions[:, 2] != 0):
            return self.render_particles_3d(positions, **kwargs)
        return self.render_particles_2d(positions, **kwargs)
