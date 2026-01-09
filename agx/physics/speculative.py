"""
Speculative Physics Engine
===========================

⚠️ THEORETICAL CONSTRUCTS ONLY ⚠️

This module implements PURELY HYPOTHETICAL physics concepts for simulation
exploration. These include:
- Negative mass/energy
- Exotic matter fields  
- Dark energy-like repulsive forces
- Alcubierre-inspired warp concepts

NONE of these represent real physics or claim to create actual effects.
All results are for EDUCATIONAL and CREATIVE exploration only.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
from agx.physics.constants import CONSTANTS


class ExoticMatterType(str, Enum):
    """Types of hypothetical exotic matter."""
    NEGATIVE_MASS = "negative_mass"
    NEGATIVE_ENERGY = "negative_energy"
    DARK_ENERGY = "dark_energy"
    PHANTOM_ENERGY = "phantom_energy"


@dataclass
class ExoticMatter:
    """
    [THEORETICAL] Exotic matter entity.
    
    ⚠️ This is a PURELY HYPOTHETICAL construct for simulation.
    Negative mass violates known physics.
    """
    id: str = ""
    matter_type: ExoticMatterType = ExoticMatterType.NEGATIVE_MASS
    mass: float = -1.0  # Negative!
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    energy_density: float = -1e-10
    stability_factor: float = 1.0  # 0-1, artificial stability constraint
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)


@dataclass
class WarpFieldConfig:
    """[THEORETICAL] Alcubierre-inspired warp field configuration."""
    bubble_radius: float = 1.0
    wall_thickness: float = 0.1
    velocity_target: float = 1.0  # Multiple of c (purely theoretical)
    energy_density_required: float = -1e30  # Negative (impossible)


class SpeculativeEngine:
    """
    [THEORETICAL] Speculative Physics Simulation Engine.

    ⚠️ ALL CONTENTS ARE HYPOTHETICAL ⚠️
    
    This engine provides simulations of speculative physics concepts:
    - Negative mass dynamics
    - Exotic energy field interactions
    - Theoretical anti-gravity effects
    - Warp field geometry (Alcubierre metric concepts)
    
    Results are for EDUCATIONAL EXPLORATION only and do not represent
    any claim of real-world feasibility.
    """
    
    THEORETICAL_DISCLAIMER = """
    ╔══════════════════════════════════════════════════════════════╗
    ║  [THEORETICAL SIMULATION]                                     ║
    ║  All results are SPECULATIVE and for EDUCATIONAL USE ONLY    ║
    ║  No claims of real-world anti-gravity effects are made       ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    def __init__(self, stability_enforcement: bool = True, max_negative_ratio: float = 0.1):
        """
        Args:
            stability_enforcement: Apply artificial constraints to prevent runaway
            max_negative_ratio: Maximum ratio of negative to positive mass
        """
        self.stability_enforcement = stability_enforcement
        self.max_negative_ratio = max_negative_ratio
        self.exotic_matter: List[ExoticMatter] = []
        self.dark_energy_density = CONSTANTS.rho_vacuum
        self._warnings: List[str] = []
    
    def add_exotic_matter(self, matter: ExoticMatter) -> bool:
        """Add exotic matter if it passes stability checks."""
        if self.stability_enforcement:
            total_positive = sum(abs(m.mass) for m in self.exotic_matter if m.mass > 0)
            total_negative = sum(abs(m.mass) for m in self.exotic_matter if m.mass < 0)
            
            if matter.mass < 0:
                new_ratio = (total_negative + abs(matter.mass)) / max(total_positive, 1)
                if new_ratio > self.max_negative_ratio:
                    self._warnings.append(f"Rejected: would exceed negative mass ratio limit")
                    return False
        
        self.exotic_matter.append(matter)
        return True
    
    def negative_mass_force(self, m1: float, m2: float, r_vec: np.ndarray) -> np.ndarray:
        """
        [THEORETICAL] Calculate force with negative mass.
        
        With negative mass, F = ma has interesting consequences:
        - Two negative masses attract BUT accelerate toward each other
        - Positive and negative mass: positive chases negative, creating runaway
        
        This uses the standard F = -Gm1m2/r² * r̂ but allows negative mass.
        """
        r_sq = np.dot(r_vec, r_vec) + 1e-10
        r = np.sqrt(r_sq)
        r_hat = r_vec / r
        
        G = CONSTANTS.G
        F_mag = G * m1 * m2 / r_sq
        
        return F_mag * r_hat
    
    def dark_energy_repulsion(self, position: np.ndarray, 
                             center: np.ndarray = np.zeros(3)) -> np.ndarray:
        """
        [THEORETICAL] Dark energy-like repulsive force.
        
        Models cosmological constant as a local repulsive effect.
        F ∝ r (increases with distance, like Λ-driven expansion)
        """
        r_vec = position - center
        r = np.linalg.norm(r_vec)
        if r < 1e-10:
            return np.zeros(3)
        
        # Hubble-like acceleration: a = H² * r
        H = 2.2e-18  # Hubble parameter
        acceleration = H**2 * r
        
        return acceleration * r_vec / r
    
    def warp_bubble_metric_factor(self, position: np.ndarray, 
                                  config: WarpFieldConfig) -> float:
        """
        [THEORETICAL] Alcubierre warp bubble shape function.
        
        f(r) = (tanh(σ(r+R)) - tanh(σ(r-R))) / (2*tanh(σR))
        
        This defines the geometry of a theoretical warp bubble.
        """
        r = np.linalg.norm(position)
        R = config.bubble_radius
        sigma = 1 / config.wall_thickness
        
        numerator = np.tanh(sigma * (r + R)) - np.tanh(sigma * (r - R))
        denominator = 2 * np.tanh(sigma * R)
        
        return numerator / denominator if abs(denominator) > 1e-10 else 0.0
    
    def warp_field_energy_density(self, position: np.ndarray, 
                                  config: WarpFieldConfig) -> float:
        """
        [THEORETICAL] Energy density required for warp bubble.
        
        The Alcubierre metric requires NEGATIVE energy density,
        which is why it's purely theoretical.
        """
        r = np.linalg.norm(position)
        R = config.bubble_radius
        sigma = 1 / config.wall_thickness
        v = config.velocity_target * CONSTANTS.c
        
        # Simplified energy density formula (actual is more complex)
        # ρ ∝ -v²σ² / (8πG) at the bubble walls
        
        G = CONSTANTS.G
        c = CONSTANTS.c
        
        # Shape function derivative contribution
        f = self.warp_bubble_metric_factor(position, config)
        df_dr = sigma * (1 - f**2) if abs(r) > 1e-10 else 0
        
        energy = -(v**2 * sigma**2 * df_dr**2) / (8 * np.pi * G * c**4)
        
        return energy
    
    def antigravity_efficiency_metric(self, force_produced: float, 
                                      energy_consumed: float) -> float:
        """
        [THEORETICAL] Calculate hypothetical anti-gravity efficiency.
        
        This is a made-up metric for optimization purposes in simulations.
        η = F_produced / E_consumed (with arbitrary scaling)
        """
        if energy_consumed <= 0:
            return 0.0
        
        return abs(force_produced) / abs(energy_consumed) * 1e10
    
    def generate_exotic_field(self, size: Tuple[float, float], 
                             resolution: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate 2D exotic energy field for visualization."""
        x = np.linspace(-size[0]/2, size[0]/2, resolution)
        y = np.linspace(-size[1]/2, size[1]/2, resolution)
        X, Y = np.meshgrid(x, y)
        
        field = np.zeros_like(X)
        
        for matter in self.exotic_matter:
            pos = matter.position[:2]
            r = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2) + 0.1
            field += matter.energy_density / r
        
        return X, Y, field
    
    def stability_analysis(self) -> Dict[str, Any]:
        """Analyze stability of current exotic matter configuration."""
        total_mass = sum(m.mass for m in self.exotic_matter)
        total_energy = sum(m.energy_density for m in self.exotic_matter)
        
        positive_mass = sum(m.mass for m in self.exotic_matter if m.mass > 0)
        negative_mass = sum(abs(m.mass) for m in self.exotic_matter if m.mass < 0)
        
        ratio = negative_mass / max(positive_mass, 1e-30)
        
        return {
            "total_mass": total_mass,
            "total_energy": total_energy,
            "positive_mass": positive_mass,
            "negative_mass": negative_mass,
            "negative_ratio": ratio,
            "is_stable": ratio <= self.max_negative_ratio,
            "warnings": self._warnings.copy(),
            "disclaimer": self.THEORETICAL_DISCLAIMER,
        }
    
    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "exotic_matter_count": len(self.exotic_matter),
            "stability": self.stability_analysis(),
            "theoretical_note": "All results are SPECULATIVE simulations only",
        }
