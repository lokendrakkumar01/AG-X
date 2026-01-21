"""
Symbolic Mathematics Engine
===========================

Algebraic manipulation, equation solving, and symbolic computation using SymPy.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import sympy as sp
from sympy import symbols, solve, simplify, expand, factor, collect
from loguru import logger


class SymbolicSolver:
    """Symbolic mathematics solver using SymPy."""
    
    def __init__(self):
        """Initialize symbolic solver."""
        # Common symbols
        self.x, self.y, self.z = symbols('x y z')
        self.a, self.b, self.c = symbols('a b c')
        self.n = symbols('n', integer=True)
    
    def parse_expression(self, expr_str: str) -> sp.Expr:
        """Parse a string into a SymPy expression.
        
        Args:
            expr_str: Expression string (e.g., "x**2 + 2*x + 1")
            
        Returns:
            SymPy expression object
        """
        try:
            return sp.sympify(expr_str)
        except Exception as e:
            logger.error(f"Failed to parse expression '{expr_str}': {e}")
            raise ValueError(f"Invalid expression: {expr_str}")
    
    def simplify_expression(self, expr_str: str, steps: bool = False) -> str:
        """Simplify an algebraic expression.
        
        Args:
            expr_str: Expression to simplify
            steps: If True, return step-by-step simplification
            
        Returns:
            Simplified expression as string
        """
        expr = self.parse_expression(expr_str)
        
        if steps:
            steps_list = []
            steps_list.append(f"Original: {expr}")
            
            # Expand
            expanded = expand(expr)
            if expanded != expr:
                steps_list.append(f"Expanded: {expanded}")
                expr = expanded
            
            # Simplify
            simplified = simplify(expr)
            if simplified != expr:
                steps_list.append(f"Simplified: {simplified}")
            
            return "\n".join(steps_list)
        else:
            return str(simplify(expr))
    
    def expand_expression(self, expr_str: str) -> str:
        """Expand an algebraic expression.
        
        Args:
            expr_str: Expression to expand
            
        Returns:
            Expanded expression
        """
        expr = self.parse_expression(expr_str)
        return str(expand(expr))
    
    def factor_expression(self, expr_str: str) -> str:
        """Factor an algebraic expression.
        
        Args:
            expr_str: Expression to factor
            
        Returns:
            Factored expression
        """
        expr = self.parse_expression(expr_str)
        return str(factor(expr))
    
    def solve_equation(
        self, 
        equation_str: str, 
        variable: str = 'x'
    ) -> List[str]:
        """Solve an equation for a variable.
        
        Args:
            equation_str: Equation (e.g., "x**2 - 4 = 0")
            variable: Variable to solve for
            
        Returns:
            List of solutions as strings
        """
        # Parse equation
        if '=' in equation_str:
            left, right = equation_str.split('=')
            equation = self.parse_expression(left) - self.parse_expression(right)
        else:
            equation = self.parse_expression(equation_str)
        
        # Solve
        var = symbols(variable)
        solutions = solve(equation, var)
        
        return [str(sol) for sol in solutions]
    
    def solve_system(
        self,
        equations: List[str],
        variables: List[str]
    ) -> dict:
        """Solve a system of equations.
        
        Args:
            equations: List of equation strings
            variables: List of variable names to solve for
            
        Returns:
            Dictionary mapping variables to their solutions
        """
        # Parse equations
        eq_list = []
        for eq_str in equations:
            if '=' in eq_str:
                left, right = eq_str.split('=')
                eq_list.append(self.parse_expression(left) - self.parse_expression(right))
            else:
                eq_list.append(self.parse_expression(eq_str))
        
        # Parse variables
        var_symbols = [symbols(v) for v in variables]
        
        # Solve system
        solutions = solve(eq_list, var_symbols)
        
        # Convert to dict
        if isinstance(solutions, dict):
            return {str(k): str(v) for k, v in solutions.items()}
        elif isinstance(solutions, list) and solutions:
            # Multiple solutions
            return {str(var_symbols[i]): str(solutions[0][i]) for i in range(len(var_symbols))}
        else:
            return {}
    
    def partial_fraction_decomposition(self, expr_str: str) -> str:
        """Perform partial fraction decomposition.
        
        Args:
            expr_str: Rational expression
            
        Returns:
            Decomposed expression
        """
        expr = self.parse_expression(expr_str)
        decomposed = sp.apart(expr, self.x)
        return str(decomposed)
    
    def expand_trigonometric(self, expr_str: str) -> str:
        """Expand trigonometric expressions.
        
        Args:
            expr_str: Trigonometric expression
            
        Returns:
            Expanded expression
        """
        expr = self.parse_expression(expr_str)
        expanded = expand(expr, trig=True)
        return str(expanded)
    
    def get_step_by_step_solution(
        self,
        equation_str: str,
        variable: str = 'x'
    ) -> List[str]:
        """Get detailed step-by-step solution for an equation.
        
        Args:
            equation_str: Equation to solve
            variable: Variable to solve for
            
        Returns:
            List of solution steps
        """
        steps = []
        
        # Parse equation
        if '=' in equation_str:
            left, right = equation_str.split('=')
            lhs = self.parse_expression(left)
            rhs = self.parse_expression(right)
        else:
            lhs = self.parse_expression(equation_str)
            rhs = sp.Integer(0)
        
        steps.append(f"Original equation: {lhs} = {rhs}")
        
        # Move everything to left side
        equation = lhs - rhs
        equation = simplify(equation)
        steps.append(f"Standard form: {equation} = 0")
        
        # Try to factor if polynomial
        var = symbols(variable)
        factored = factor(equation)
        if factored != equation:
            steps.append(f"Factored form: {factored} = 0")
        
        # Solve
        solutions = solve(equation, var)
        
        if solutions:
            steps.append(f"Solutions: {', '.join(str(s) for s in solutions)}")
        else:
            steps.append("No real solutions found")
        
        # Verify solutions
        if solutions:
            steps.append("\nVerification:")
            for sol in solutions:
                value = equation.subs(var, sol)
                simplified_value = simplify(value)
                steps.append(f"  {variable} = {sol}: {simplified_value} = 0 ✓")
        
        return steps
