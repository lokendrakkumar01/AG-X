# AG-X 2026 Deployment Guide 🚀

## Quick Hosting Options

### Option 1: Render.com (Recommended - Free Tier)

1. **Create account** at [render.com](https://render.com)
2. **Connect GitHub** repository
3. **Create Web Service**:
   - Build Command: `pip install -e .`
   - Start Command: `gunicorn agx.viz.dashboard:server --bind 0.0.0.0:$PORT`
4. **Deploy** - Your app will be live at `https://agx-2026.onrender.com`

### Option 2: Railway.app (Easy)

1. **Visit** [railway.app](https://railway.app)
2. **New Project** → Deploy from GitHub
3. **Select** your AG-X repository
4. Railway auto-detects Python and deploys
5. **Get URL** from dashboard

### Option 3: PythonAnywhere (Free)

1. **Create account** at [pythonanywhere.com](https://pythonanywhere.com)
2. **Upload** project files
3. **Create Web App** → Flask
4. **Configure WSGI**:
   ```python
   import sys
   sys.path.insert(0, '/home/username/AG-X')
   from agx.viz.dashboard import server as application
   ```

### Option 4: Heroku

```bash
# Install Heroku CLI
heroku login
heroku create agx-2026
git push heroku main
heroku open
```

### Option 5: Docker (Any Cloud)

```bash
# Build and run
docker build -t agx-2026 .
docker run -p 8050:8050 agx-2026

# Push to Docker Hub
docker tag agx-2026 yourusername/agx-2026
docker push yourusername/agx-2026
```

---

## Environment Variables

Set these in your hosting platform:

| Variable | Value | Description |
|----------|-------|-------------|
| `AGX_ENV` | `production` | Environment mode |
| `AGX_DEBUG` | `false` | Disable debug |
| `PORT` | `8050` | Server port (auto-set by most hosts) |

---

## Files Created for Hosting

| File | Platform |
|------|----------|
| `Procfile` | Heroku, Railway |
| `render.yaml` | Render.com |
| `railway.json` | Railway.app |
| `requirements.txt` | All platforms |
| `runtime.txt` | Python version |
| `Dockerfile` | Docker/Kubernetes |

---

## Local Development

```bash
# Run locally
cd AG-X
pip install -e .
python -m agx.viz.dashboard

# Open http://localhost:8050
```

---

⚠️ **Note**: Free hosting tiers may have cold starts (app sleeps after inactivity). First request may take 30-60 seconds to wake up.
