# AG-X 2026 Production Configuration
# ====================================
# For deployment on Render, Railway, or any cloud platform

web: gunicorn agx.viz.dashboard:server --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120
