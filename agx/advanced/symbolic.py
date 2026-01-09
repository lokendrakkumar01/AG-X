"""
Symbolic Mathematics Integration
=================================

Equation discovery, symbolic manipulation, and LaTeX rendering.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import sympy as sp
from sympy import symbols, Function, Eq, solve, diff, integrate, simplify, latex


@dataclass
class DiscoveredEquation:
    """A discovered or derived equation."""
    expression: sp.Expr
    variables: List[sp.Symbol]
    r_squared: float = 0.0
    complexity: int = 0
    latex_form: str = ""
    description: str = ""


class SymbolicMath:
    """
    Symbolic Mathematics Engine.
    
    Provides equation discovery from data, symbolic manipulation,
    derivation of physics equations, and LaTeX rendering.
    """
    
    # Common physics symbols
    PHYSICS_SYMBOLS = {
        "G": sp.Symbol("G", positive=True),      # Gravitational constant
        "c": sp.Symbol("c", positive=True),      # Speed of light
        "m": sp.Symbol("m", positive=True),      # Mass
        "M": sp.Symbol("M", positive=True),      # Large mass
        "r": sp.Symbol("r", positive=True),      # Distance
        "t": sp.Symbol("t", real=True),          # Time
        "v": sp.Symbol("v", real=True),          # Velocity
        "a": sp.Symbol("a", real=True),          # Acceleration
        "F": sp.Symbol("F", real=True),          # Force
        "E": sp.Symbol("E", real=True),          # Energy
        "rho": sp.Symbol("rho", real=True),      # Density
    }
    
    def __init__(self):
        # Make symbols available as attributes
        for name, sym in self.PHYSICS_SYMBOLS.items():
            setattr(self, name, sym)
    
    def newton_gravity_law(self) -> sp.Expr:
        """F = GMm/r²"""
        G, M, m, r = self.G, self.M, self.m, self.r
        return G * M * m / r**2
    
    def schwarzschild_radius(self) -> sp.Expr:
        """r_s = 2GM/c²"""
        G, M, c = self.G, self.M, self.c
        return 2 * G * M / c**2
    
    def escape_velocity(self) -> sp.Expr:
        """v_esc = √(2GM/r)"""
        G, M, r = self.G, self.M, self.r
        return sp.sqrt(2 * G * M / r)
    
    def time_dilation_factor(self) -> sp.Expr:
        """√(1 - r_s/r) = √(1 - 2GM/(rc²))"""
        G, M, r, c = self.G, self.M, self.r, self.c
        return sp.sqrt(1 - 2 * G * M / (r * c**2))
    
    def fit_polynomial(self, 
                      x_data: np.ndarray, 
                      y_data: np.ndarray,
                      max_degree: int = 5) -> DiscoveredEquation:
        """Fit polynomial to data and return symbolic expression."""
        x = sp.Symbol('x')
        best_eq = None
        best_r2 = -np.inf
        
        for degree in range(1, max_degree + 1):
            coeffs = np.polyfit(x_data, y_data, degree)
            poly = np.poly1d(coeffs)
            y_pred = poly(x_data)
            
            ss_res = np.sum((y_data - y_pred)**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            if r2 > best_r2 and r2 > 0.8:  # Only keep if good fit
                best_r2 = r2
                # Build symbolic polynomial
                expr = sum(c * x**i for i, c in enumerate(reversed(coeffs)))
                best_eq = DiscoveredEquation(
                    expression=simplify(expr),
                    variables=[x],
                    r_squared=r2,
                    complexity=degree,
                    latex_form=latex(simplify(expr)),
                )
        
        return best_eq
    
    def discover_power_law(self,
                          x_data: np.ndarray,
                          y_data: np.ndarray) -> DiscoveredEquation:
        """Discover power law relationship y = a * x^b."""
        x = sp.Symbol('x')
        a, b = sp.symbols('a b')
        
        # Fit in log space
        mask = (x_data > 0) & (y_data > 0)
        if np.sum(mask) < 2:
            return None
        
        log_x = np.log(x_data[mask])
        log_y = np.log(y_data[mask])
        
        coeffs = np.polyfit(log_x, log_y, 1)
        b_val = coeffs[0]
        a_val = np.exp(coeffs[1])
        
        # R² calculation
        y_pred = a_val * x_data[mask]**b_val
        ss_res = np.sum((y_data[mask] - y_pred)**2)
        ss_tot = np.sum((y_data[mask] - np.mean(y_data[mask]))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        expr = sp.Float(a_val, 4) * x**sp.Float(b_val, 4)
        
        return DiscoveredEquation(
            expression=expr,
            variables=[x],
            r_squared=r2,
            complexity=2,
            latex_form=latex(expr),
            description=f"Power law: y = {a_val:.4f} * x^{b_val:.4f}",
        )
    
    def derive_equation(self, expr: sp.Expr, var: sp.Symbol) -> sp.Expr:
        """Take derivative of expression."""
        return diff(expr, var)
    
    def integrate_equation(self, expr: sp.Expr, var: sp.Symbol) -> sp.Expr:
        """Integrate expression."""
        return integrate(expr, var)
    
    def solve_equation(self, equation: sp.Expr, var: sp.Symbol) -> List[sp.Expr]:
        """Solve equation for variable."""
        return solve(equation, var)
    
    def substitute_values(self, expr: sp.Expr, values: Dict[str, float]) -> float:
        """Substitute numerical values into expression."""
        subs = {}
        for name, val in values.items():
            if hasattr(self, name):
                subs[getattr(self, name)] = val
        return float(expr.subs(subs))
    
    def to_latex(self, expr: sp.Expr) -> str:
        """Convert expression to LaTeX string."""
        return latex(expr)
    
    def to_python_function(self, expr: sp.Expr, variables: List[sp.Symbol]):
        """Convert symbolic expression to Python function."""
        from sympy import lambdify
        return lambdify(variables, expr, modules=['numpy'])
    
    def analyze_dimensions(self, expr: sp.Expr) -> str:
        """Simple dimensional analysis hint."""
        expr_str = str(expr)
        
        if 'G' in expr_str and 'M' in expr_str:
            if 'c' in expr_str:
                return "Likely relativistic expression"
            return "Gravitational expression"
        elif 'c' in expr_str:
            return "Relativistic/electromagnetic expression"
        return "General expression"
    
    def generate_physics_report(self, equations: List[DiscoveredEquation]) -> str:
        """Generate markdown report of discovered equations."""
        lines = ["# Discovered Equations\n"]
        
        for i, eq in enumerate(equations, 1):
            lines.append(f"## Equation {i}")
            lines.append(f"\n$$\n{eq.latex_form}\n$$\n")
            lines.append(f"- **R²:** {eq.r_squared:.4f}")
            lines.append(f"- **Complexity:** {eq.complexity}")
            if eq.description:
                lines.append(f"- **Description:** {eq.description}")
            lines.append("")
        
        return "\n".join(lines)
