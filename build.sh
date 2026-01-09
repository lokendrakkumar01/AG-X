# Render.com Build Script
# ========================

set -e

echo "Installing AG-X 2026..."
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn

echo "Build complete!"
