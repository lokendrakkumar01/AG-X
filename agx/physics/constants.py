"""
Physical Constants and Units System
====================================

Comprehensive physics constants with unit management for multi-scale simulations.
All values are in SI units unless otherwise specified.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum


class UnitSystem(str, Enum):
    """Supported unit systems."""
    SI = "si"                    # Standard SI units
    CGS = "cgs"                  # Centimeter-gram-second
    NATURAL = "natural"          # c = ℏ = G = 1
    GEOMETRIC = "geometric"      # c = G = 1
    PLANCK = "planck"           # All Planck units = 1


@dataclass(frozen=True)
class PhysicsConstants:
    """
    Fundamental physics constants.
    
    All values are in SI units. Constants marked with [THEORETICAL] are
    speculative values used only for simulation exploration.
    """
    
    # Gravitational Constants
    G: float = 6.67430e-11          # Gravitational constant [m³/(kg·s²)]
    
    # Electromagnetic Constants  
    c: float = 299792458.0          # Speed of light [m/s]
    epsilon_0: float = 8.8541878e-12  # Vacuum permittivity [F/m]
    mu_0: float = 1.25663706e-6     # Vacuum permeability [H/m]
    
    # Quantum Constants
    h: float = 6.62607015e-34       # Planck constant [J·s]
    hbar: float = 1.054571817e-34   # Reduced Planck constant [J·s]
    
    # Particle Physics
    m_e: float = 9.1093837e-31      # Electron mass [kg]
    m_p: float = 1.67262192e-27     # Proton mass [kg]
    e: float = 1.60217663e-19       # Elementary charge [C]
    
    # Thermodynamic
    k_B: float = 1.380649e-23       # Boltzmann constant [J/K]
    
    # Planck Units (derived)
    @property
    def l_P(self) -> float:
        """Planck length [m]"""
        return np.sqrt(self.hbar * self.G / self.c**3)
    
    @property
    def t_P(self) -> float:
        """Planck time [s]"""
        return np.sqrt(self.hbar * self.G / self.c**5)
    
    @property
    def m_P(self) -> float:
        """Planck mass [kg]"""
        return np.sqrt(self.hbar * self.c / self.G)
    
    @property
    def E_P(self) -> float:
        """Planck energy [J]"""
        return np.sqrt(self.hbar * self.c**5 / self.G)
    
    # Cosmological Constants
    H_0: float = 2.2e-18            # Hubble constant [1/s] (~70 km/s/Mpc)
    
    # =========================================================================
    # THEORETICAL/SPECULATIVE CONSTANTS
    # These are hypothetical values for simulation exploration only!
    # =========================================================================
    
    # [THEORETICAL] Vacuum energy density (much smaller than QFT prediction)
    rho_vacuum: float = 5.96e-27    # [kg/m³] - cosmological observation
    
    # [THEORETICAL] Dark energy equation of state parameter
    w_dark_energy: float = -1.0     # w = p/ρ for cosmological constant
    
    # [THEORETICAL] Hypothetical exotic matter coupling
    lambda_exotic: float = 1e-10    # Coupling strength (dimensionless)


@dataclass
class Units:
    """
    Unit conversion and management system.
    
    Provides conversion factors between different unit systems and
    manages simulation unit scaling.
    """
    
    system: UnitSystem = UnitSystem.SI
    
    # Custom scaling factors for simulation
    length_scale: float = 1.0       # meters per simulation unit
    time_scale: float = 1.0         # seconds per simulation unit
    mass_scale: float = 1.0         # kg per simulation unit
    
    _constants: PhysicsConstants = field(default_factory=PhysicsConstants)
    
    def __post_init__(self):
        """Initialize derived scales."""
        self._velocity_scale = self.length_scale / self.time_scale
        self._acceleration_scale = self.length_scale / self.time_scale**2
        self._force_scale = self.mass_scale * self._acceleration_scale
        self._energy_scale = self.mass_scale * self.length_scale**2 / self.time_scale**2
    
    @classmethod
    def astronomical(cls) -> "Units":
        """Create units suitable for astronomical simulations."""
        return cls(
            system=UnitSystem.SI,
            length_scale=1.496e11,   # 1 AU in meters
            time_scale=3.154e7,      # 1 year in seconds
            mass_scale=1.989e30,     # 1 solar mass in kg
        )
    
    @classmethod
    def laboratory(cls) -> "Units":
        """Create units suitable for laboratory-scale simulations."""
        return cls(
            system=UnitSystem.SI,
            length_scale=1e-3,       # millimeters
            time_scale=1e-6,         # microseconds
            mass_scale=1e-6,         # micrograms
        )
    
    @classmethod
    def planck(cls) -> "Units":
        """Create Planck units (natural units where everything is ~1)."""
        const = PhysicsConstants()
        return cls(
            system=UnitSystem.PLANCK,
            length_scale=const.l_P,
            time_scale=const.t_P,
            mass_scale=const.m_P,
        )
    
    def to_si(self, value: float, unit_type: str) -> float:
        """Convert from simulation units to SI."""
        scales = {
            "length": self.length_scale,
            "time": self.time_scale,
            "mass": self.mass_scale,
            "velocity": self._velocity_scale,
            "acceleration": self._acceleration_scale,
            "force": self._force_scale,
            "energy": self._energy_scale,
        }
        return value * scales.get(unit_type, 1.0)
    
    def from_si(self, value: float, unit_type: str) -> float:
        """Convert from SI to simulation units."""
        scales = {
            "length": self.length_scale,
            "time": self.time_scale,
            "mass": self.mass_scale,
            "velocity": self._velocity_scale,
            "acceleration": self._acceleration_scale,
            "force": self._force_scale,
            "energy": self._energy_scale,
        }
        return value / scales.get(unit_type, 1.0)
    
    @property
    def G_sim(self) -> float:
        """Gravitational constant in simulation units."""
        # G [m³/(kg·s²)] -> simulation units
        return (self._constants.G * 
                self.time_scale**2 * self.mass_scale / self.length_scale**3)
    
    @property
    def c_sim(self) -> float:
        """Speed of light in simulation units."""
        return self._constants.c * self.time_scale / self.length_scale


# Default instances
CONSTANTS = PhysicsConstants()
SI_UNITS = Units(system=UnitSystem.SI)
ASTRO_UNITS = Units.astronomical()
PLANCK_UNITS = Units.planck()


def get_schwarzschild_radius(mass: float, units: Units = SI_UNITS) -> float:
    """
    Calculate Schwarzschild radius for a given mass.
    
    r_s = 2GM/c²
    
    Args:
        mass: Mass in simulation units
        units: Unit system to use
    
    Returns:
        Schwarzschild radius in simulation units
    """
    mass_si = units.to_si(mass, "mass")
    r_s_si = 2 * CONSTANTS.G * mass_si / CONSTANTS.c**2
    return units.from_si(r_s_si, "length")


def get_escape_velocity(mass: float, radius: float, units: Units = SI_UNITS) -> float:
    """
    Calculate escape velocity from a massive body.
    
    v_esc = √(2GM/r)
    
    Args:
        mass: Mass in simulation units
        radius: Distance from center in simulation units
        units: Unit system to use
    
    Returns:
        Escape velocity in simulation units
    """
    mass_si = units.to_si(mass, "mass")
    radius_si = units.to_si(radius, "length")
    v_esc_si = np.sqrt(2 * CONSTANTS.G * mass_si / radius_si)
    return units.from_si(v_esc_si, "velocity")
