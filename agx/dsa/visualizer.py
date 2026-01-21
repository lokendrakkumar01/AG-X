"""
Algorithm Visualizer
====================

Visual representations of algorithms and data structures.
"""

from __future__ import annotations

from typing import List
import plotly.graph_objects as go
import numpy as np


class AlgorithmVisualizer:
    """Visualize algorithms step-by-step."""
    
    @staticmethod
    def visualize_array(arr: List[int], title: str = "Array", highlights: List[int] = None) -> go.Figure:
        """Visualize an array as a bar chart.
        
        Args:
            arr: Array to visualize
            title: Plot title
            highlights: Indices to highlight
            
        Returns:
            Plotly Figure
        """
        highlights = highlights or []
        colors = ['red' if i in highlights else 'blue' for i in range(len(arr))]
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(range(len(arr))),
                y=arr,
                marker_color=colors,
                text=arr,
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Index",
            yaxis_title="Value",
            showlegend=False,
            height=400
        )
        
        return fig
    
    @staticmethod
    def visualize_sorting_steps(steps: List[List[int]]) -> go.Figure:
        """Visualize sorting algorithm steps.
        
        Args:
            steps: List of array states at each step
            
        Returns:
            Plotly Figure with animation
        """
        frames = []
        
        for i, arr in enumerate(steps):
            frames.append(go.Frame(
                data=[go.Bar(
                    x=list(range(len(arr))),
                    y=arr,
                    marker_color='blue',
                    text=arr,
                    textposition='outside'
                )],
                name=str(i)
            ))
        
        fig = go.Figure(
            data=[go.Bar(
                x=list(range(len(steps[0]))),
                y=steps[0],
                marker_color='blue',
                text=steps[0],
                textposition='outside'
            )],
            layout=go.Layout(
                title="Sorting Algorithm Visualization",
                updatemenus=[{
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 500}, "fromcurrent": True}],
                            "label": "Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                            "label": "Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }],
                sliders=[{
                    "active": 0,
                    "steps": [{"args": [[f.name]], "label": f.name, "method": "animate"} for f in frames],
                    "x": 0.1,
                    "len": 0.9,
                    "xanchor": "left",
                    "y": 0,
                    "yanchor": "top"
                }]
            ),
            frames=frames
        )
        
        return fig
    
    @staticmethod
    def visualize_binary_tree(nodes: List[int]) -> str:
        """Create ASCII visualization of binary tree.
        
        Args:
            nodes: Array representation of tree (level-order)
            
        Returns:
            ASCII art string
        """
        if not nodes:
            return "Empty tree"
        
        # Simple ASCII tree (basic implementation)
        lines = []
        lines.append("Binary Tree (Level-order):")
        lines.append(f"Root: {nodes[0]}")
        
        if len(nodes) > 1:
            lines.append(f"Level 1: {nodes[1:3]}")
        if len(nodes) > 3:
            lines.append(f"Level 2: {nodes[3:7]}")
        if len(nodes) > 7:
            lines.append(f"Level 3: {nodes[7:15]}")
        
        return "\n".join(lines)
    
    @staticmethod
    def explain_complexity(time_complexity: str, space_complexity: str) -> str:
        """Explain time and space complexity.
        
        Args:
            time_complexity: Time complexity (e.g., "O(n)")
            space_complexity: Space complexity
            
        Returns:
            Explanation string
        """
        explanations = {"O(1)": "Constant - doesn't depend on input size",
                       "O(log n)": "Logarithmic - divides problem in half each time",
                       "O(n)": "Linear - grows proportionally with input",
                       "O(n log n)": "Linearithmic - efficient sorting algorithms",
                       "O(n^2)": "Quadratic - nested loops over input",
                       "O(n^3)": "Cubic - triple nested loops",
                       "O(2^n)": "Exponential - very slow for large inputs",
                       "O(n!)": "Factorial - extremely slow, avoid if possible"}
        
        time_exp = explanations.get(time_complexity, "Custom complexity")
        space_exp = explanations.get(space_complexity, "Custom complexity")
        
        return (
            f"**Time Complexity: {time_complexity}**\n"
            f"{time_exp}\n\n"
            f"**Space Complexity: {space_complexity}**\n"
            f"{space_exp}"
        )
