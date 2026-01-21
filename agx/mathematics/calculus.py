"""
Calculus Solver
===============

Derivatives, integrals, limits, and differential equations.
"""

from __future__ import annotations

from typing import Optional
import sympy as sp
from sympy import symbols, diff, integrate, limit, oo, dsolve
from loguru import logger


class CalculusSolver:
    """Calculus operations solver."""
    
    def __init__(self):
        """Initialize calculus solver."""
        self.x = symbols('x')
        self.y = symbols('y', cls=sp.Function)
    
    def parse_expression(self, expr_str: str) -> sp.Expr:
        """Parse string to SymPy expression."""
        return sp.sympify(expr_str)
    
    def derivative(
        self,
        expr_str: str,
        variable: str = 'x',
        order: int = 1,
        show_steps: bool = False
    ) -> str:
        """Calculate derivative of an expression.
        
        Args:
            expr_str: Expression to differentiate
            variable: Variable to differentiate with respect to
            order: Order of derivative
            show_steps: Show step-by-step derivation
            
        Returns:
            Derivative as string
        """
        expr = self.parse_expression(expr_str)
        var = symbols(variable)
        
        if show_steps and order == 1:
            steps = []
            steps.append(f"Original function: f({variable}) = {expr}")
            
            # Try to identify rule
            derivative = diff(expr, var)
            steps.append(f"Derivative: f'({variable}) = {derivative}")
            
            # Simplified
            simplified = sp.simplify(derivative)
            if simplified != derivative:
                steps.append(f"Simplified: f'({variable}) = {simplified}")
            
            return "\n".join(steps)
        else:
            derivative = diff(expr, var, order)
            return str(sp.simplify(derivative))
    
    def integral(
        self,
        expr_str: str,
        variable: str = 'x',
        definite: bool = False,
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None
    ) -> str:
        """Calculate integral of an expression.
        
        Args:
            expr_str: Expression to integrate
            variable: Variable to integrate with respect to
            definite: If True, calculate definite integral
            lower_limit: Lower bound for definite integral
            upper_limit: Upper bound for definite integral
            
        Returns:
            Integral as string
        """
        expr = self.parse_expression(expr_str)
        var = symbols(variable)
        
        if definite and lower_limit is not None and upper_limit is not None:
            # Definite integral
            result = integrate(expr, (var, lower_limit, upper_limit))
            return str(sp.simplify(result))
        else:
            # Indefinite integral
            result = integrate(expr, var)
            return str(result) + " + C"
    
    def limit_calc(
        self,
        expr_str: str,
        variable: str = 'x',
        point: str = '0',
        direction: str = 'both'
    ) -> str:
        """Calculate limit of an expression.
        
        Args:
            expr_str: Expression
            variable: Variable
            point: Point to approach (can be 'oo' for infinity)
            direction: Direction ('both', '+', '-')
            
        Returns:
            Limit as string
        """
        expr = self.parse_expression(expr_str)
        var = symbols(variable)
        
        # Parse point
        if point.lower() == 'oo' or point == '∞':
            pt = oo
        elif point.lower() == '-oo':
            pt = -oo
        else:
            pt = sp.sympify(point)
        
        # Calculate limit
        if direction == '+':
            result = limit(expr, var, pt, '+')
        elif direction == '-':
            result = limit(expr, var, pt, '-')
        else:
            result = limit(expr, var, pt)
        
        return str(result)
    
    def taylor_series(
        self,
        expr_str: str,
        variable: str = 'x',
        point: float = 0,
        order: int = 5
    ) -> str:
        """Calculate Taylor series expansion.
        
        Args:
            expr_str: Expression to expand
            variable: Variable
            point: Point of expansion
            order: Order of expansion
            
        Returns:
            Taylor series as string
        """
        expr = self.parse_expression(expr_str)
        var = symbols(variable)
        
        series = sp.series(expr, var, point, order + 1).removeO()
        return str(series)
    
    def critical_points(
        self,
        expr_str: str,
        variable: str = 'x'
    ) -> dict:
        """Find critical points (local max/min/inflection).
        
        Args:
            expr_str: Expression
            variable: Variable
            
        Returns:
            Dictionary with critical point information
        """
        expr = self.parse_expression(expr_str)
        var = symbols(variable)
        
        # First derivative (for critical points)
        first_deriv = diff(expr, var)
        critical_pts = sp.solve(first_deriv, var)
        
        # Second derivative (for classification)
        second_deriv = diff(first_deriv, var)
        
        results = {
            "critical_points": [],
            "local_maxima": [],
            "local_minima": [],
            "inconclusive": []
        }
        
        for pt in critical_pts:
            if pt.is_real:
                results["critical_points"].append(str(pt))
                
                # Second derivative test
                second_deriv_value = second_deriv.subs(var, pt)
                
                if second_deriv_value > 0:
                    results["local_minima"].append(str(pt))
                elif second_deriv_value < 0:
                    results["local_maxima"].append(str(pt))
                else:
                    results["inconclusive"].append(str(pt))
        
        return results
    
    def solve_ode(
        self,
        ode_str: str,
        function: str = 'y',
        variable: str = 'x'
    ) -> str:
        """Solve ordinary differential equation.
        
        Args:
            ode_str: ODE string (use y for function, x for variable)
            function: Function name
            variable: Independent variable
            
        Returns:
            Solution as string
        """
        var = symbols(variable)
        func = sp.Function(function)
        
        # Parse ODE
        ode = self.parse_expression(ode_str)
        
        # Solve
        solution = dsolve(ode, func(var))
        
        return str(solution)
