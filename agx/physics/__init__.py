"""
AG-X 2026 Physics Module
========================

Multi-scale gravity modeling combining classical mechanics, general relativity
approximations, quantum-inspired fields, and speculative physics constructs.

All speculative physics models are clearly labeled as THEORETICAL.
"""

from agx.physics.constants import PhysicsConstants, Units
from agx.physics.newtonian import NewtonianEngine, Body, GravitationalForce
from agx.physics.general_relativity import GREngine, SpacetimeCurvature
from agx.physics.quantum_field import QuantumFieldEngine, VacuumFluctuation
from agx.physics.speculative import SpeculativeEngine, ExoticMatter
from agx.physics.solver import NumericalSolver, SimulationState
from agx.physics.engine import PhysicsEngine

__all__ = [
    # Constants
    "PhysicsConstants",
    "Units",
    # Newtonian
    "NewtonianEngine",
    "Body",
    "GravitationalForce",
    # General Relativity
    "GREngine",
    "SpacetimeCurvature",
    # Quantum Field
    "QuantumFieldEngine",
    "VacuumFluctuation",
    # Speculative
    "SpeculativeEngine",
    "ExoticMatter",
    # Solver
    "NumericalSolver",
    "SimulationState",
    # Main Engine
    "PhysicsEngine",
]
