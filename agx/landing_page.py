"""
AG-X 2026 Universal Platform - Main Landing Page
================================================

Landing page showcasing all educational modules.
"""

from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

def create_landing_page():
    """Create landing page layout."""
    return dbc.Container([
        # Header
        html.Div([
            html.H1("🚀 AG-X 2026", className="display-3 text-center fw-bold", 
                   style={"color": "#00D9FF"}),
            html.P("Universal Knowledge, Programming, DSA & Collaborative Learning System",
                  className="lead text-center text-muted mb-4"),
        ], className="text-center py-5"),
        
        # Disclaimer
        dbc.Alert([
            html.Strong("⚠️ Educational Disclaimer: "),
            "All simulations, calculations, and models are for EDUCATIONAL PURPOSES ONLY."
        ], color="warning", className="mb-5"),
        
        # Module Cards
        dbc.Row([
            # Physics
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("🔬 Physics", className="mb-0")),
                    dbc.CardBody([
                        html.P("Advanced Gravity Research & Simulation", className="text-muted"),
                        html.Ul([
                            html.Li("Newtonian Mechanics & N-body simulation"),
                            html.Li("General Relativity approximations"),
                            html.Li("Quantum field concepts"),
                            html.Li("Speculative physics (theoretical)"),
                        ]),
                        dbc.Button("Launch Physics Dashboard", 
                                 href="/physics", color="primary", className="mt-2"),
                    ]),
                ], className="mb-4 shadow"),
            ], md=6),
            
            # Chemistry
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("🧪 Chemistry", className="mb-0")),
                    dbc.CardBody([
                        html.P("Chemical Equations & Molecular Tools", className="text-muted"),
                        html.Ul([
                            html.Li("Equation balancer (matrix method)"),
                            html.Li("3D molecular visualization"),
                            html.Li("Thermodynamics calculator"),
                            html.Li("Chemical kinetics"),
                        ]),
                        dbc.Button("Explore Chemistry", 
                                 href="/chemistry", color="success", className="mt-2"),
                    ]),
                ], className="mb-4 shadow"),
            ], md=6),
        ]),
        
        dbc.Row([
            # Mathematics
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("📐 Mathematics", className="mb-0")),
                    dbc.CardBody([
                        html.P("Symbolic Computation & Graphing", className="text-muted"),
                        html.Ul([
                            html.Li("Equation solver (algebraic, systems)"),
                            html.Li("Calculus (derivatives, integrals)"),
                            html.Li("Interactive 2D/3D graphing"),
                            html.Li("Step-by-step solutions"),
                        ]),
                        dbc.Button("Solve Math Problems", 
                                 href="/mathematics", color="info", className="mt-2"),
                    ]),
                ], className="mb-4 shadow"),
            ], md=6),
            
            # DSA Practice
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("💻 DSA Practice", className="mb-0")),
                    dbc.CardBody([
                        html.P("Data Structures & Algorithms", className="text-muted"),
                        html.Ul([
                            html.Li("Problem bank (Arrays, Trees, Graphs, DP)"),
                            html.Li("Python code execution"),
                            html.Li("Algorithm visualizations"),
                            html.Li("Complexity analysis"),
                        ]),
                        dbc.Button("Practice DSA", 
                                 href="/dsa", color="danger", className="mt-2"),
                    ]),
                ], className="mb-4 shadow"),
            ], md=6),
        ]),
        
        dbc.Row([
            # Programming
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("🌐 Programming", className="mb-0")),
                    dbc.CardBody([
                        html.P("12+ Languages Knowledge Base", className="text-muted"),
                        html.Ul([
                            html.Li("Python, Java, C++, JavaScript, Go, Rust"),
                            html.Li("Language tutorials & examples"),
                            html.Li("Best practices & patterns"),
                            html.Li("Use cases & comparisons"),
                        ]),
                        dbc.Button("Learn Programming", 
                                 href="/programming", color="warning", className="mt-2"),
                    ]),
                ], className="mb-4 shadow"),
            ], md=6),
            
            # English
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("🗣️ English Communication", className="mb-0")),
                    dbc.CardBody([
                        html.P("Vocabulary & Conversation Practice", className="text-muted"),
                        html.Ul([
                            html.Li("Daily vocabulary with pronunciation"),
                            html.Li("Conversation scenarios"),
                            html.Li("Interview preparation"),
                            html.Li("Professional communication"),
                        ]),
                        dbc.Button("Improve English", 
                                 href="/english", color="secondary", className="mt-2"),
                    ]),
                ], className="mb-4 shadow"),
            ], md=6),
        ]),
        
        # Quick Start
        html.Hr(className="my-5"),
        html.H3("🚀 Quick Start", className="text-center mb-4"),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("1. Choose a Module", className="text-primary"),
                    html.P("Select from Physics, Chemistry, Math, DSA, Programming, or English above"),
                ], className="text-center p-3"),
            ], md=4),
            dbc.Col([
                html.Div([
                    html.H5("2. Explore & Learn", className="text-success"),
                    html.P("Use interactive tools, solve problems, and practice skills"),
                ], className="text-center p-3"),
            ], md=4),
            dbc.Col([
                html.Div([
                    html.H5("3. Track Progress",  className="text-info"),
                    html.P("Save work, review solutions, and improve continuously"),
                ], className="text-center p-3"),
            ], md=4),
        ]),
        
        # Footer
        html.Hr(className="my-4"),
        html.Div([
            html.P([
                "AG-X 2026 | ",
                html.A("GitHub", href="https://github.com/lokendrakkumar01/AG-X", 
                      target="_blank", className="text-info me-3"),
                " | ",
                html.A("Documentation", href="#", className="text-info"),
            ], className="text-center text-muted"),
            html.P("Built with Python, Dash, SymPy, Plotly", 
                  className="text-center text-muted small"),
        ], className="py-4"),
        
    ], fluid=True, className="py-4")


def create_app():
    """Create Dash application."""
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title="AG-X 2026 - Universal Educational Platform",
    )
    
    app.layout = create_landing_page()
    return app


# For production deployment
server = create_app().server


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=8050)
