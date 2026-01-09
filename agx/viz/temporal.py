"""
Temporal Visualization Module
==============================

4D temporal evolution visualization, animations, and phase space plots.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class AnimationConfig:
    """Animation configuration."""
    fps: int = 30
    duration_ms: int = 100
    transition_ms: int = 50
    loop: bool = True


class TemporalVisualizer:
    """
    Temporal Evolution Visualizer.
    
    Creates animations, phase space trajectories, and time evolution plots.
    """
    
    def __init__(self, config: Optional[AnimationConfig] = None):
        self.config = config or AnimationConfig()
    
    def create_animation(self,
                        trajectory: np.ndarray,
                        times: Optional[np.ndarray] = None,
                        title: str = "Simulation Evolution") -> go.Figure:
        """
        Create animated visualization of particle trajectories.
        
        Args:
            trajectory: Shape (timesteps, n_particles, 3)
            times: Time values for each frame
        """
        n_frames = len(trajectory)
        n_particles = trajectory.shape[1]
        
        if times is None:
            times = np.arange(n_frames)
        
        # Initial frame
        fig = go.Figure(
            data=[go.Scatter3d(
                x=trajectory[0, :, 0],
                y=trajectory[0, :, 1],
                z=trajectory[0, :, 2],
                mode="markers+lines",
                marker=dict(size=8, color=np.arange(n_particles), colorscale="Plasma"),
            )],
            layout=go.Layout(
                title=f"{title} - t = {times[0]:.2f}",
                scene=dict(
                    xaxis=dict(range=[trajectory[:,:,0].min()-1, trajectory[:,:,0].max()+1]),
                    yaxis=dict(range=[trajectory[:,:,1].min()-1, trajectory[:,:,1].max()+1]),
                    zaxis=dict(range=[trajectory[:,:,2].min()-1, trajectory[:,:,2].max()+1]),
                    aspectmode="cube",
                ),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=0.1,
                    buttons=[
                        dict(label="▶ Play", method="animate",
                             args=[None, {"frame": {"duration": self.config.duration_ms},
                                         "transition": {"duration": self.config.transition_ms}}]),
                        dict(label="⏸ Pause", method="animate",
                             args=[[None], {"frame": {"duration": 0},
                                           "mode": "immediate"}]),
                    ]
                )],
            ),
        )
        
        # Animation frames
        frames = []
        for i in range(0, n_frames, max(1, n_frames // 100)):  # Limit frames
            frames.append(go.Frame(
                data=[go.Scatter3d(
                    x=trajectory[i, :, 0],
                    y=trajectory[i, :, 1],
                    z=trajectory[i, :, 2],
                    mode="markers",
                    marker=dict(size=8, color=np.arange(n_particles), colorscale="Plasma"),
                )],
                name=str(i),
                layout=go.Layout(title=f"{title} - t = {times[i]:.2f}"),
            ))
        
        fig.frames = frames
        
        # Add slider
        fig.update_layout(
            sliders=[dict(
                active=0,
                steps=[dict(method="animate", args=[[f.name]], label=f.name) for f in frames],
                x=0.1, len=0.8,
                currentvalue=dict(prefix="Frame: "),
            )],
            template="plotly_dark",
        )
        
        return fig
    
    def plot_phase_space(self,
                        positions: np.ndarray,
                        velocities: np.ndarray,
                        particle_idx: int = 0,
                        dimensions: Tuple[int, int] = (0, 1)) -> go.Figure:
        """
        Plot phase space trajectory for a particle.
        
        Args:
            positions: Shape (timesteps, n_particles, 3)
            velocities: Shape (timesteps, n_particles, 3)
            particle_idx: Which particle to plot
            dimensions: Which position/velocity dimensions
        """
        d1, d2 = dimensions
        x = positions[:, particle_idx, d1]
        v = velocities[:, particle_idx, d1]
        
        # Color by time
        t = np.arange(len(x))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x, y=v,
            mode="lines+markers",
            marker=dict(size=4, color=t, colorscale="Viridis", colorbar=dict(title="Time")),
            line=dict(width=1, color="rgba(100, 200, 255, 0.5)"),
        ))
        
        dim_labels = ["x", "y", "z"]
        fig.update_layout(
            title=f"Phase Space - Particle {particle_idx}",
            xaxis_title=f"Position {dim_labels[d1]}",
            yaxis_title=f"Velocity {dim_labels[d1]}",
            template="plotly_dark",
        )
        
        return fig
    
    def plot_energy_evolution(self,
                             times: np.ndarray,
                             kinetic: np.ndarray,
                             potential: np.ndarray) -> go.Figure:
        """Plot energy components over time."""
        total = kinetic + potential
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=("Energy Components", "Energy Conservation"))
        
        # Energy components
        fig.add_trace(go.Scatter(x=times, y=kinetic, name="Kinetic", 
                                line=dict(color="#ff6b6b")), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=potential, name="Potential",
                                line=dict(color="#4ecdc4")), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=total, name="Total",
                                line=dict(color="#ffe66d", width=3)), row=1, col=1)
        
        # Energy error
        error = (total - total[0]) / np.abs(total[0]) * 100
        fig.add_trace(go.Scatter(x=times, y=error, name="Error %",
                                line=dict(color="#ff9f43")), row=2, col=1)
        
        fig.update_layout(
            title="Energy Evolution",
            template="plotly_dark",
            height=600,
        )
        fig.update_yaxes(title_text="Energy", row=1, col=1)
        fig.update_yaxes(title_text="Relative Error (%)", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        
        return fig
    
    def plot_lyapunov_evolution(self,
                               times: np.ndarray,
                               separations: np.ndarray) -> go.Figure:
        """Plot trajectory separation for chaos visualization."""
        fig = go.Figure()
        
        # Log of separation
        log_sep = np.log(np.maximum(separations, 1e-15))
        
        fig.add_trace(go.Scatter(
            x=times, y=log_sep,
            mode="lines",
            name="log(separation)",
            line=dict(color="#ff6b6b", width=2),
        ))
        
        # Linear fit for Lyapunov exponent
        coeffs = np.polyfit(times, log_sep, 1)
        fit_line = coeffs[0] * times + coeffs[1]
        
        fig.add_trace(go.Scatter(
            x=times, y=fit_line,
            mode="lines",
            name=f"λ ≈ {coeffs[0]:.4f}",
            line=dict(color="#4ecdc4", width=2, dash="dash"),
        ))
        
        fig.update_layout(
            title="Lyapunov Exponent Estimation",
            xaxis_title="Time",
            yaxis_title="log(δ)",
            template="plotly_dark",
        )
        
        return fig
    
    def plot_time_series(self,
                        times: np.ndarray,
                        data: Dict[str, np.ndarray],
                        title: str = "Time Series") -> go.Figure:
        """Plot multiple time series on the same graph."""
        fig = go.Figure()
        
        colors = ["#ff6b6b", "#4ecdc4", "#ffe66d", "#a29bfe", "#fd79a8"]
        
        for i, (name, values) in enumerate(data.items()):
            fig.add_trace(go.Scatter(
                x=times, y=values,
                name=name,
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_dark",
            legend=dict(x=0.02, y=0.98),
        )
        
        return fig
