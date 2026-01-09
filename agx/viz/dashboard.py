"""
Enhanced Interactive Web Dashboard
===================================

Real-time Dash-based dashboard with comprehensive simulation controls.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import json

from dash import Dash, html, dcc, callback, Output, Input, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    title: str = "AG-X 2026 - Gravity Research Platform"
    update_interval: int = 1000
    theme: str = "dark"
    debug: bool = False


def create_header() -> html.Div:
    """Create dashboard header."""
    return html.Div([
        html.Div([
            html.H1("🚀 AG-X 2026", className="display-4 text-primary mb-0"),
            html.P("Advanced Gravity Research Simulation Platform", className="lead text-muted"),
        ], className="text-center"),
        html.Hr(className="my-3"),
        dbc.Alert([
            html.Strong("⚠️ Scientific Disclaimer: "),
            "All simulations are THEORETICAL and for EDUCATIONAL purposes only. ",
            "No claims of real anti-gravity effects are made."
        ], color="warning", className="mb-3 text-center"),
    ], className="py-4")


def create_simulation_panel() -> dbc.Card:
    """Create main simulation control panel."""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-rocket me-2"),
            html.Strong("🎛️ Simulation Configuration"),
        ]),
        dbc.CardBody([
            dbc.Row([
                # Scenario Selection
                dbc.Col([
                    html.Label("Scenario", className="fw-bold text-info"),
                    dcc.Dropdown(
                        id="scenario-dropdown",
                        options=[
                            {"label": "🌍 Two-Body System", "value": "two_body"},
                            {"label": "🔺 Three-Body Problem", "value": "three_body"},
                            {"label": "☀️ Solar System", "value": "solar_system"},
                            {"label": "🌀 Binary Star", "value": "binary_star"},
                            {"label": "🌌 Galaxy Collision", "value": "galaxy_collision"},
                            {"label": "⚫ Black Hole Orbit", "value": "black_hole"},
                            {"label": "🔴 Negative Mass [THEORY]", "value": "negative_mass"},
                            {"label": "⭕ Warp Field [THEORY]", "value": "warp_field"},
                        ],
                        value="two_body",
                        className="mb-3",
                    ),
                ], md=4),
                
                # Physics Model Selection
                dbc.Col([
                    html.Label("Physics Models", className="fw-bold text-info"),
                    dcc.Checklist(
                        id="physics-checklist",
                        options=[
                            {"label": " Newtonian Gravity", "value": "newtonian"},
                            {"label": " GR Corrections", "value": "gr"},
                            {"label": " Quantum Fields", "value": "quantum"},
                            {"label": " Speculative [THEORY]", "value": "speculative"},
                        ],
                        value=["newtonian"],
                        className="mb-3",
                        inputClassName="me-2",
                    ),
                ], md=4),
                
                # Integration Method
                dbc.Col([
                    html.Label("Solver Method", className="fw-bold text-info"),
                    dcc.Dropdown(
                        id="solver-dropdown",
                        options=[
                            {"label": "Leapfrog (Symplectic)", "value": "leapfrog"},
                            {"label": "RK4 (Classic)", "value": "rk4"},
                            {"label": "DOP853 (High Precision)", "value": "dop853"},
                            {"label": "Adaptive RK45", "value": "rk45"},
                        ],
                        value="leapfrog",
                        className="mb-3",
                    ),
                ], md=4),
            ]),
            
            html.Hr(),
            
            dbc.Row([
                # Timesteps
                dbc.Col([
                    html.Label("Timesteps", className="fw-bold"),
                    dcc.Slider(
                        id="timesteps-slider",
                        min=100, max=10000, step=100,
                        value=1000,
                        marks={100: "100", 1000: "1K", 5000: "5K", 10000: "10K"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], md=3),
                
                # Time Step (dt)
                dbc.Col([
                    html.Label("Time Step (dt)", className="fw-bold"),
                    dcc.Slider(
                        id="dt-slider",
                        min=0.001, max=0.1, step=0.001,
                        value=0.01,
                        marks={0.001: "0.001", 0.01: "0.01", 0.05: "0.05", 0.1: "0.1"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], md=3),
                
                # Number of Bodies
                dbc.Col([
                    html.Label("Number of Bodies", className="fw-bold"),
                    dcc.Slider(
                        id="bodies-slider",
                        min=2, max=100, step=1,
                        value=2,
                        marks={2: "2", 10: "10", 50: "50", 100: "100"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], md=3),
                
                # Softening Length
                dbc.Col([
                    html.Label("Softening (ε)", className="fw-bold"),
                    dcc.Slider(
                        id="softening-slider",
                        min=-8, max=-2, step=1,
                        value=-6,
                        marks={-8: "1e-8", -6: "1e-6", -4: "1e-4", -2: "1e-2"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], md=3),
            ], className="mb-4"),
            
            html.Hr(),
            
            # Action Buttons
            dbc.Row([
                dbc.Col([
                    dbc.Button("▶️ Run Simulation", id="run-btn", color="success", size="lg", className="me-2"),
                    dbc.Button("⏸️ Pause", id="pause-btn", color="warning", size="lg", className="me-2"),
                    dbc.Button("🔄 Reset", id="reset-btn", color="secondary", size="lg", className="me-2"),
                    dbc.Button("📊 Export Report", id="report-btn", color="info", size="lg", className="me-2"),
                    dbc.Button("💾 Save State", id="save-btn", color="primary", size="lg"),
                ], className="text-center"),
            ]),
        ])
    ], className="shadow mb-4", style={"backgroundColor": "#1e2a3a"})


def create_advanced_options() -> dbc.Accordion:
    """Create advanced options accordion."""
    return dbc.Accordion([
        dbc.AccordionItem([
            dbc.Row([
                dbc.Col([
                    html.Label("Central Mass (Solar Masses)", className="fw-bold"),
                    dbc.Input(id="central-mass", type="number", value=1.0, min=0.01, max=1000, step=0.1),
                ], md=3),
                dbc.Col([
                    html.Label("Orbital Radius (AU)", className="fw-bold"),
                    dbc.Input(id="orbital-radius", type="number", value=1.0, min=0.1, max=100, step=0.1),
                ], md=3),
                dbc.Col([
                    html.Label("Initial Velocity Factor", className="fw-bold"),
                    dbc.Input(id="velocity-factor", type="number", value=1.0, min=0.1, max=2.0, step=0.1),
                ], md=3),
                dbc.Col([
                    html.Label("Eccentricity", className="fw-bold"),
                    dbc.Input(id="eccentricity", type="number", value=0.0, min=0.0, max=0.99, step=0.01),
                ], md=3),
            ]),
        ], title="⚙️ Orbital Parameters"),
        
        dbc.AccordionItem([
            dbc.Row([
                dbc.Col([
                    dbc.Checklist(
                        id="gr-options",
                        options=[
                            {"label": " Time Dilation Effects", "value": "time_dilation"},
                            {"label": " Perihelion Precession", "value": "precession"},
                            {"label": " Frame Dragging", "value": "frame_drag"},
                            {"label": " Gravitational Waves", "value": "gw"},
                        ],
                        value=[],
                        className="mb-2",
                    ),
                ], md=6),
                dbc.Col([
                    html.Label("Curvature Resolution", className="fw-bold"),
                    dcc.Slider(id="curvature-res", min=16, max=128, step=16, value=64,
                              marks={16: "16", 64: "64", 128: "128"}),
                ], md=6),
            ]),
        ], title="🌀 General Relativity Options"),
        
        dbc.AccordionItem([
            dbc.Row([
                dbc.Col([
                    dbc.Checklist(
                        id="speculative-options",
                        options=[
                            {"label": " Negative Mass Particles", "value": "negative_mass"},
                            {"label": " Exotic Energy Fields", "value": "exotic_energy"},
                            {"label": " Dark Energy Repulsion", "value": "dark_energy"},
                            {"label": " Warp Bubble Geometry", "value": "warp_bubble"},
                            {"label": " Casimir Effect", "value": "casimir"},
                        ],
                        value=[],
                        className="mb-2",
                    ),
                ], md=6),
                dbc.Col([
                    html.Label("Negative Mass Ratio", className="fw-bold"),
                    dcc.Slider(id="neg-mass-ratio", min=0, max=0.5, step=0.05, value=0.1,
                              marks={0: "0%", 0.25: "25%", 0.5: "50%"}),
                    html.Small("⚠️ Purely theoretical construct", className="text-warning"),
                ], md=6),
            ]),
        ], title="🔮 Speculative Physics [THEORETICAL]"),
        
        dbc.AccordionItem([
            dbc.Row([
                dbc.Col([
                    html.Label("AI Optimization", className="fw-bold"),
                    dcc.Dropdown(
                        id="ai-method",
                        options=[
                            {"label": "None", "value": "none"},
                            {"label": "Bayesian Optimization", "value": "bayesian"},
                            {"label": "Reinforcement Learning (PPO)", "value": "rl"},
                            {"label": "Evolutionary (NSGA-II)", "value": "evolutionary"},
                        ],
                        value="none",
                    ),
                ], md=4),
                dbc.Col([
                    html.Label("Target Metric", className="fw-bold"),
                    dcc.Dropdown(
                        id="target-metric",
                        options=[
                            {"label": "Energy Conservation", "value": "energy"},
                            {"label": "Orbital Stability", "value": "stability"},
                            {"label": "Anti-Gravity Efficiency [THEORY]", "value": "antigrav"},
                        ],
                        value="energy",
                    ),
                ], md=4),
                dbc.Col([
                    html.Label("AI Iterations", className="fw-bold"),
                    dbc.Input(id="ai-iterations", type="number", value=100, min=10, max=1000),
                ], md=4),
            ]),
        ], title="🧠 AI Optimization Settings"),
        
    ], start_collapsed=True, className="mb-4")


def create_metrics_panel() -> dbc.Card:
    """Create real-time metrics panel."""
    return dbc.Card([
        dbc.CardHeader("📈 Real-Time Metrics"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H3("0.000", id="total-energy", className="text-warning mb-0"),
                        html.Small("Total Energy", className="text-muted"),
                    ], className="text-center p-3 rounded", style={"backgroundColor": "#2d3748"}),
                ], md=2),
                dbc.Col([
                    html.Div([
                        html.H3("0.000%", id="energy-drift", className="text-success mb-0"),
                        html.Small("Energy Drift", className="text-muted"),
                    ], className="text-center p-3 rounded", style={"backgroundColor": "#2d3748"}),
                ], md=2),
                dbc.Col([
                    html.Div([
                        html.H3("Stable", id="chaos-status", className="text-info mb-0"),
                        html.Small("System State", className="text-muted"),
                    ], className="text-center p-3 rounded", style={"backgroundColor": "#2d3748"}),
                ], md=2),
                dbc.Col([
                    html.Div([
                        html.H3("0.00", id="lyapunov-exp", className="text-primary mb-0"),
                        html.Small("Lyapunov λ", className="text-muted"),
                    ], className="text-center p-3 rounded", style={"backgroundColor": "#2d3748"}),
                ], md=2),
                dbc.Col([
                    html.Div([
                        html.H3("0", id="sim-time", className="text-light mb-0"),
                        html.Small("Sim Time", className="text-muted"),
                    ], className="text-center p-3 rounded", style={"backgroundColor": "#2d3748"}),
                ], md=2),
                dbc.Col([
                    html.Div([
                        html.H3("0 fps", id="sim-fps", className="text-secondary mb-0"),
                        html.Small("Performance", className="text-muted"),
                    ], className="text-center p-3 rounded", style={"backgroundColor": "#2d3748"}),
                ], md=2),
            ]),
        ])
    ], className="shadow mb-4", style={"backgroundColor": "#1e2a3a"})


def create_visualization_tabs() -> dbc.Tabs:
    """Create visualization tabs."""
    return dbc.Tabs([
        dbc.Tab([
            dcc.Graph(id="particles-graph", style={"height": "550px"}),
        ], label="🔵 3D Particles", tab_id="particles"),
        
        dbc.Tab([
            dcc.Graph(id="orbits-graph", style={"height": "550px"}),
        ], label="🛤️ Orbital Paths", tab_id="orbits"),
        
        dbc.Tab([
            dcc.Graph(id="spacetime-graph", style={"height": "550px"}),
        ], label="🌀 Spacetime Curvature", tab_id="spacetime"),
        
        dbc.Tab([
            dcc.Graph(id="energy-graph", style={"height": "550px"}),
        ], label="⚡ Energy Evolution", tab_id="energy"),
        
        dbc.Tab([
            dcc.Graph(id="phase-graph", style={"height": "550px"}),
        ], label="📊 Phase Space", tab_id="phase"),
        
        dbc.Tab([
            dcc.Graph(id="field-graph", style={"height": "550px"}),
        ], label="🌊 Gravity Field", tab_id="field"),
        
        dbc.Tab([
            html.Div([
                html.H4("AI Analysis Results", className="text-info mt-3"),
                html.Pre(id="ai-results", className="p-3 rounded", 
                        style={"backgroundColor": "#2d3748", "color": "#e0e0e0"}),
            ]),
        ], label="🧠 AI Insights", tab_id="ai"),
        
    ], id="viz-tabs", active_tab="particles", className="mb-4")


def create_footer() -> html.Div:
    """Create footer."""
    return html.Div([
        html.Hr(),
        html.P([
            "AG-X 2026 | Built with Python, Dash, PyTorch | ",
            html.A("Documentation", href="#", className="text-info me-2"),
            " | ",
            html.A("GitHub", href="#", className="text-info"),
        ], className="text-center text-muted"),
        html.P([
            html.Small("⚠️ All simulations are theoretical. No real anti-gravity claims are made."),
        ], className="text-center text-warning"),
    ], className="py-3")


def create_layout() -> html.Div:
    """Create complete dashboard layout."""
    return html.Div([
        dbc.Container([
            create_header(),
            create_simulation_panel(),
            create_advanced_options(),
            create_metrics_panel(),
            create_visualization_tabs(),
            create_footer(),
            
            # Hidden stores
            dcc.Interval(id="update-interval", interval=100, disabled=True),
            dcc.Store(id="simulation-state", data={}),
            dcc.Store(id="history-store", data=[]),
        ], fluid=True),
    ], style={
        "backgroundColor": "#0d1421",
        "minHeight": "100vh",
        "padding": "20px",
        "color": "#e0e0e0",
    })


class Dashboard:
    """Interactive simulation dashboard."""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.app = self._create_app()
        self._setup_callbacks()
        self.simulation_running = False
    
    def _create_app(self) -> Dash:
        """Create Dash application."""
        app = Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.CYBORG,
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
            ],
            title=self.config.title,
            suppress_callback_exceptions=True,
        )
        app.layout = create_layout()
        return app
    
    def _setup_callbacks(self) -> None:
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output("particles-graph", "figure"),
             Output("total-energy", "children"),
             Output("chaos-status", "children"),
             Output("sim-time", "children")],
            Input("run-btn", "n_clicks"),
            [State("scenario-dropdown", "value"),
             State("timesteps-slider", "value"),
             State("bodies-slider", "value"),
             State("physics-checklist", "value")],
            prevent_initial_call=True,
        )
        def run_simulation(n_clicks, scenario, timesteps, n_bodies, physics_models):
            """Run simulation and update visualization."""
            np.random.seed(42)
            
            # Generate based on scenario
            if scenario == "galaxy_collision":
                n = min(n_bodies * 5, 200)
                pos1 = np.random.randn(n//2, 3) * 2 + np.array([-5, 0, 0])
                pos2 = np.random.randn(n//2, 3) * 2 + np.array([5, 0, 0])
                positions = np.vstack([pos1, pos2])
                masses = np.random.uniform(0.1, 1, n)
            elif scenario == "black_hole":
                n = n_bodies
                angles = np.random.uniform(0, 2*np.pi, n-1)
                radii = np.random.uniform(2, 8, n-1)
                positions = np.zeros((n, 3))
                positions[1:, 0] = radii * np.cos(angles)
                positions[1:, 1] = radii * np.sin(angles)
                masses = np.ones(n)
                masses[0] = 100  # Black hole
            elif scenario == "negative_mass":
                n = n_bodies
                positions = np.random.randn(n, 3) * 3
                masses = np.random.choice([-1, 1], n) * np.random.uniform(0.5, 2, n)
            elif scenario == "warp_field":
                n = n_bodies
                positions = np.random.randn(n, 3) * 2
                positions[0] = [0, 0, 0]  # Warp center
                masses = np.ones(n)
            else:
                n = n_bodies
                positions = np.random.randn(n, 3) * 3
                masses = np.random.uniform(0.5, 2, n)
            
            # Color by mass
            colors = masses
            sizes = np.abs(masses) / np.abs(masses).max() * 15 + 5
            
            # 3D scatter
            fig = go.Figure(data=[go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=colors,
                    colorscale="Plasma" if scenario != "negative_mass" else "RdBu",
                    colorbar=dict(title="Mass"),
                    line=dict(width=1, color="white"),
                ),
                text=[f"Body {i}<br>Mass: {m:.2f}" for i, m in enumerate(masses)],
                hoverinfo="text",
            )])
            
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1421",
                plot_bgcolor="#0d1421",
                scene=dict(
                    xaxis=dict(gridcolor="#1e3a5f"),
                    yaxis=dict(gridcolor="#1e3a5f"),
                    zaxis=dict(gridcolor="#1e3a5f"),
                    aspectmode="cube",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                title=f"Scenario: {scenario.replace('_', ' ').title()}",
            )
            
            # Simulated metrics
            energy = np.random.uniform(-10, -5)
            chaos = "Chaotic ⚠️" if scenario in ["three_body", "galaxy_collision"] else "Stable ✓"
            sim_time = f"{timesteps * 0.01:.2f}s"
            
            return fig, f"{energy:.3f}", chaos, sim_time
        
        @self.app.callback(
            Output("spacetime-graph", "figure"),
            Input("run-btn", "n_clicks"),
            State("scenario-dropdown", "value"),
            prevent_initial_call=True,
        )
        def update_spacetime(n_clicks, scenario):
            """Update spacetime curvature visualization."""
            x = np.linspace(-10, 10, 80)
            y = np.linspace(-10, 10, 80)
            X, Y = np.meshgrid(x, y)
            
            if scenario == "black_hole":
                Z = -5 / (np.sqrt(X**2 + Y**2) + 0.5)
            elif scenario == "binary_star":
                Z = -2 / (np.sqrt((X-2)**2 + Y**2) + 0.5) - 2 / (np.sqrt((X+2)**2 + Y**2) + 0.5)
            elif scenario == "warp_field":
                r = np.sqrt(X**2 + Y**2)
                Z = np.tanh(3-r) * np.exp(-r**2/20)
            else:
                Z = -np.exp(-(X**2 + Y**2) / 8)
            
            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale="Viridis",
                opacity=0.9,
            )])
            
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1421",
                scene=dict(
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
                    zaxis_title="Curvature Depth",
                ),
                title="Spacetime Curvature (Embedding Diagram)",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            
            return fig
        
        @self.app.callback(
            Output("energy-graph", "figure"),
            Input("run-btn", "n_clicks"),
            State("timesteps-slider", "value"),
            prevent_initial_call=True,
        )
        def update_energy(n_clicks, timesteps):
            """Update energy evolution plot."""
            t = np.linspace(0, timesteps * 0.01, timesteps)
            
            # Simulated energy data
            kinetic = 5 * np.exp(-t/20) * (1 + 0.1 * np.sin(5*t))
            potential = -10 + 3 * (1 - np.exp(-t/20))
            total = kinetic + potential
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=("Energy Components", "Conservation Error"))
            
            fig.add_trace(go.Scatter(x=t, y=kinetic, name="Kinetic", 
                                    line=dict(color="#ff6b6b", width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=potential, name="Potential",
                                    line=dict(color="#4ecdc4", width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=total, name="Total",
                                    line=dict(color="#ffe66d", width=3)), row=1, col=1)
            
            error = (total - total[0]) / np.abs(total[0]) * 100
            fig.add_trace(go.Scatter(x=t, y=error, name="Error %",
                                    line=dict(color="#ff9f43", width=2)), row=2, col=1)
            
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1421",
                height=500,
                title="Energy Conservation Analysis",
            )
            
            return fig
    
    def run(self, host: str = "0.0.0.0", port: int = 8050, debug: bool = None):
        """Run the dashboard server."""
        debug = debug if debug is not None else self.config.debug
        self.app.run(host=host, port=port, debug=debug)


def create_app() -> Dash:
    """Create and return Dash app for deployment."""
    dashboard = Dashboard()
    return dashboard.app


# For Gunicorn/production deployment
server = create_app().server


if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run(debug=True)
