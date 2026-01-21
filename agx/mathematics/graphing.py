"""
Mathematical Graphing System
============================

2D and 3D plotting for functions, parametric curves, and more.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


class GraphPlotter:
    """Create interactive mathematical plots."""
    
    def __init__(self):
        """Initialize graph plotter."""
        pass
    
    def plot_function(
        self,
        expr_str: str,
        x_range: Tuple[float, float] = (-10, 10),
        num_points: int = 1000,
        title: Optional[str] = None
    ) -> go.Figure:
        """Plot a 2D function.
        
        Args:
            expr_str: Function expression (e.g., "x**2 + 2*x")
            x_range: (min, max) for x-axis
            num_points: Number of points to plot
            title: Plot title
            
        Returns:
            Plotly Figure
        """
        # Parse expression
        x_sym = sp.Symbol('x')
        expr = parse_expr(expr_str)
        
        # Convert to numpy function
        f = sp.lambdify(x_sym, expr, 'numpy')
        
        # Generate points
        x = np.linspace(x_range[0], x_range[1], num_points)
        try:
            y = f(x)
        except Exception as e:
            # Handle division by zero, etc.
            y = np.full_like(x, np.nan)
            for i, xi in enumerate(x):
                try:
                    y[i] = float(f(xi))
                except:
                    y[i] = np.nan
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=f"f(x) = {expr_str}",
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=title or f"Plot of f(x) = {expr_str}",
            xaxis_title="x",
            yaxis_title="y",
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def plot_multiple_functions(
        self,
        functions: List[Tuple[str, str]],  # List of (expression, label)
        x_range: Tuple[float, float] = (-10, 10),
        num_points: int = 1000
    ) -> go.Figure:
        """Plot multiple functions on the same axes.
        
        Args:
            functions: List of (expression, label) tuples
            x_range: (min, max) for x-axis
            num_points: Number of points to plot
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        x = np.linspace(x_range[0], x_range[1], num_points)
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        
        for idx, (expr_str, label) in enumerate(functions):
            # Parse and evaluate
            x_sym = sp.Symbol('x')
            expr = parse_expr(expr_str)
            f = sp.lambdify(x_sym, expr, 'numpy')
            
            try:
                y = f(x)
            except:
                y = np.full_like(x, np.nan)
                for i, xi in enumerate(x):
                    try:
                        y[i] = float(f(xi))
                    except:
                        y[i] = np.nan
            
            color = colors[idx % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=label,
                line=dict(color=color, width=2)
            ))
        
        fig.update_layout(
            title="Multiple Functions",
            xaxis_title="x",
            yaxis_title="y",
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def plot_3d_surface(
        self,
        expr_str: str,
        x_range: Tuple[float, float] = (-5, 5),
        y_range: Tuple[float, float] = (-5, 5),
        num_points: int = 50
    ) -> go.Figure:
        """Plot a 3D surface.
        
        Args:
            expr_str: Function expression with x and y (e.g., "x**2 + y**2")
            x_range: (min, max) for x-axis
            y_range: (min, max) for y-axis
            num_points: Number of points per axis
            
        Returns:
            Plotly Figure
        """
        # Parse expression
        x_sym, y_sym = sp.symbols('x y')
        expr = parse_expr(expr_str)
        
        # Convert to numpy function
        f = sp.lambdify((x_sym, y_sym), expr, 'numpy')
        
        # Generate mesh
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)
        X, Y = np.meshgrid(x, y)
        
        try:
            Z = f(X, Y)
        except:
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        Z[i, j] = float(f(X[i, j], Y[i, j]))
                    except:
                        Z[i, j] = np.nan
        
        # Create plot
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
        
        fig.update_layout(
            title=f"Surface plot: z = {expr_str}",
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z',
            ),
            height=600
        )
        
        return fig
    
    def plot_parametric(
        self,
        x_expr: str,
        y_expr: str,
        t_range: Tuple[float, float] = (0, 10),
        num_points: int = 1000
    ) -> go.Figure:
        """Plot a parametric curve.
        
        Args:
            x_expr: Expression for x(t)
            y_expr: Expression for y(t)
            t_range: (min, max) for parameter t
            num_points: Number of points
            
        Returns:
            Plotly Figure
        """
        t_sym = sp.Symbol('t')
        
        x_func = sp.lambdify(t_sym, parse_expr(x_expr), 'numpy')
        y_func = sp.lambdify(t_sym, parse_expr(y_expr), 'numpy')
        
        t = np.linspace(t_range[0], t_range[1], num_points)
        x = x_func(t)
        y = y_func(t)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=f"Parametric curve: x(t)={x_expr}, y(t)={y_expr}",
            xaxis_title="x",
            yaxis_title="y",
            height=500
        )
        
        return fig
