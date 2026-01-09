"""
General Relativity Approximation Engine
========================================

Simplified GR effects: spacetime curvature, time dilation, geodesics.
⚠️ These are APPROXIMATIONS for education, not full numerical relativity.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from agx.physics.constants import CONSTANTS


@dataclass
class SpacetimeCurvature:
    """Spacetime curvature at a point."""
    position: np.ndarray
    metric: np.ndarray
    time_dilation: float = 1.0
    ricci_scalar: float = 0.0


@dataclass
class GRBody:
    """A massive body curving spacetime."""
    mass: float
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    spin: float = 0.0
    
    @property
    def schwarzschild_radius(self) -> float:
        return 2 * CONSTANTS.G * self.mass / CONSTANTS.c**2


class GREngine:
    """
    General Relativity Approximation Engine.
    
    Provides metric tensors, time dilation, geodesics, and curvature visualization.
    """
    
    def __init__(self, c: float = CONSTANTS.c, G: float = CONSTANTS.G):
        self.c, self.G = c, G
        self.bodies: List[GRBody] = []
    
    def add_body(self, body: GRBody) -> None:
        self.bodies.append(body)
    
    def minkowski_metric(self) -> np.ndarray:
        return np.diag([-self.c**2, 1.0, 1.0, 1.0])
    
    def get_metric_at_point(self, position: np.ndarray) -> np.ndarray:
        """Calculate metric tensor at position (weak-field approximation)."""
        g = self.minkowski_metric()
        for body in self.bodies:
            r = np.linalg.norm(position - body.position)
            if r < 1e-10: continue
            phi = -self.G * body.mass / r
            h = 2 * phi / self.c**2
            g[0, 0] += h * self.c**2
            g[1, 1] -= h; g[2, 2] -= h; g[3, 3] -= h
        return g
    
    def gravitational_time_dilation(self, position: np.ndarray) -> float:
        """Time dilation factor (< 1 near massive objects)."""
        g_00 = self.get_metric_at_point(position)[0, 0]
        return np.sqrt(-g_00 / self.c**2) if g_00 < 0 else 0.0
    
    def curvature_grid(self, center: np.ndarray = np.zeros(3), size: float = 10.0, 
                       resolution: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate 2D curvature grid for visualization."""
        x = np.linspace(center[0] - size/2, center[0] + size/2, resolution)
        y = np.linspace(center[1] - size/2, center[1] + size/2, resolution)
        X, Y = np.meshgrid(x, y)
        curvature = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                pos = np.array([X[i,j], Y[i,j], center[2]])
                curvature[i,j] = 1.0 - self.gravitational_time_dilation(pos)
        return X, Y, curvature
    
    def geodesic_acceleration(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Geodesic acceleration with 1PN corrections."""
        accel = np.zeros(3)
        for body in self.bodies:
            r_vec = position - body.position
            r = np.linalg.norm(r_vec)
            if r < 1e-10: continue
            r_hat = r_vec / r
            a_newton = -self.G * body.mass / r**2
            v2 = np.dot(velocity, velocity)
            phi = -self.G * body.mass / r
            pn_factor = 1 + (v2/self.c**2 - 2*phi/self.c**2)
            accel += a_newton * pn_factor * r_hat
        return accel
    
    def light_deflection_angle(self, impact_parameter: float, mass: float) -> float:
        """Einstein's light deflection: α = 4GM/(bc²)."""
        return 4 * self.G * mass / (impact_parameter * self.c**2) if impact_parameter > 0 else np.inf
    
    def gravitational_redshift(self, emit_pos: np.ndarray, obs_pos: np.ndarray) -> float:
        """Redshift z between positions."""
        d_emit = self.gravitational_time_dilation(emit_pos)
        d_obs = self.gravitational_time_dilation(obs_pos)
        return d_obs / d_emit - 1 if d_emit > 0 and d_obs > 0 else np.inf
    
    def embedding_diagram_height(self, r: float, mass: float) -> float:
        """Height z(r) for Schwarzschild embedding diagram."""
        r_s = 2 * self.G * mass / self.c**2
        return 2 * np.sqrt(r_s * (r - r_s)) if r > r_s else np.inf
    
    def generate_embedding_surface(self, mass: float, r_max: float = 10.0, 
                                   resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate 3D embedding diagram surface."""
        r_s = 2 * self.G * mass / self.c**2
        r = np.linspace(1.1 * r_s, r_max, resolution)
        theta = np.linspace(0, 2*np.pi, resolution)
        R, Theta = np.meshgrid(r, theta)
        Z = np.where(R > r_s, 2 * np.sqrt(r_s * (R - r_s)), np.nan)
        return R * np.cos(Theta), R * np.sin(Theta), Z
