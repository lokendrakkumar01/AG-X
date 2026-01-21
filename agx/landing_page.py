"""
AG-X 2026 - Home Page
=====================
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Create app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="AG-X 2026 - Universal Educational Platform",
    suppress_callback_exceptions=True
)

# Layout
app.layout = dbc.Container([
    # Header
    html.Div([
        html.H1("🚀 AG-X 2026", className="display-3 text-center text-primary"),
        html.P("Universal Educational Platform", className="lead text-center"),
    ], className="py-5 text-center"),
    
    # Alert
    dbc.Alert([
        html.Strong("⚠️ Educational Platform: "),
        "All tools are for learning purposes only."
    ], color="warning", className="mb-4"),
    
    # Modules Grid
    html.H3("📚 Available Modules", className="text-center mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("🔬 Physics")),
                dbc.CardBody([
                    html.P("Gravity simulations & mechanics"),
                    html.Ul([
                        html.Li("Newtonian mechanics"),
                        html.Li("General relativity"),
                        html.Li("Quantum fields"),
                    ]),
                    dbc.Button("Launch Physics", href="/physics", color="primary"),
                ]),
            ], className="mb-3 shadow"),
        ], md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("🧪 Chemistry")),
                dbc.CardBody([
                    html.P("Chemical equations & molecules"),
                    html.Ul([
                        html.Li("Equation balancer"),
                        html.Li("3D molecular viz"),
                        html.Li("Thermodynamics"),
                    ]),
                    dbc.Button("Launch Chemistry", href="/chemistry", color="success"),
                ]),
            ], className="mb-3 shadow"),
        ], md=6),
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("📐 Mathematics")),
                dbc.CardBody([
                    html.P("Symbolic computation & graphing"),
                    html.Ul([
                        html.Li("Equation solver"),
                        html.Li("Calculus tools"),
                        html.Li("Interactive graphs"),
                    ]),
                    dbc.Button("Launch Math", href="/math", color="info"),
                ]),
            ], className="mb-3 shadow"),
        ], md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("💻 DSA Practice")),
                dbc.CardBody([
                    html.P("Coding problems & algorithms"),
                    html.Ul([
                        html.Li("Problem bank"),
                        html.Li("Code execution"),
                        html.Li("Visualizations"),
                    ]),
                    dbc.Button("Practice DSA", href="/dsa", color="danger"),
                ]),
            ], className="mb-3 shadow"),
        ], md=6),
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("🌐 Programming")),
                dbc.CardBody([
                    html.P("12+ languages, tutorials & examples"),
                    dbc.Button("Learn Programming", href="/programming", color="warning"),
                ]),
            ], className="mb-3 shadow"),
        ], md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("🗣️ English")),
                dbc.CardBody([
                    html.P("Vocabulary & conversation practice"),
                    dbc.Button("Improve English", href="/english", color="secondary"),
                ]),
            ], className="mb-3 shadow"),
        ], md=6),
    ]),
    
    # Footer
    html.Hr(className="my-5"),
    html.Div([
        html.P("AG-X 2026 | Python + Dash + SymPy + Plotly", className="text-center text-muted"),
        html.P([
            html.A("GitHub", href="https://github.com/lokendrakkumar01/AG-X", 
                  target="_blank", className="text-info"),
        ], className="text-center"),
    ], className="pb-4"),
    
], fluid=True, style={"backgroundColor": "#0d1421", "minHeight": "100vh", "color": "#e0e0e0"})

# Export server for Gunicorn
server = app.server

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
