"""
AG-X 2026 Mathematics Module
=============================

Symbolic computation, calculus, linear algebra, and mathematical problem solving.
"""

from .symbolic import SymbolicSolver
from .calculus import CalculusSolver  
from .graphing import GraphPlotter

__all__ = [
    "SymbolicSolver",
    "CalculusSolver",
    "GraphPlotter",
]
