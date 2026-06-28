# Dockerfile
# Job: Package the entire fraud detection API into a container.
# This means Python version, all libraries, all code, all outputs — everything.
# Wherever this container runs, it behaves exactly the same.
# Build: docker build -t fraud-detection .
# Run:   docker run -p 8000:8000 fraud-detection

# ── 1. Base Image ─────────────────────────────────────────────────────────────
# We start from an official Python 3.11 image.
# Think of this as a clean Ubuntu machine with Python already installed.
# slim-bookworm = lightweight version, no unnecessary tools, smaller container size.
FROM python:3.11-slim-bookworm

# ── 2. Set Working Directory ──────────────────────────────────────────────────
# All commands from here run inside /app inside the container.
# This is the container's equivalent of your ML_Project_3 folder.
WORKDIR /app

# ── 3. Install System Dependencies ───────────────────────────────────────────
# Some Python libraries need system-level packages to compile.
# We install them first before installing Python packages.
# --no-install-recommends = don't install optional extras, keeps image small.
# rm -rf /var/lib/apt/lists/* = clean up apt cache after installing, saves space.
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# ── 4. Copy and Install Requirements ─────────────────────────────────────────
# We copy requirements.txt FIRST before copying the rest of the code.
# Why? Docker caches each step. If requirements.txt hasn't changed,
# Docker skips reinstalling libraries even if your code changed.
# This makes rebuilds much faster during development.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 5. Copy Source Code ───────────────────────────────────────────────────────
# Now copy everything else — src, api, outputs (model, scaler, threshold).
# .dockerignore (created separately) tells Docker what to skip.
COPY src/ ./src/
COPY api/ ./api/
COPY outputs/ ./outputs/

# ── 6. Expose Port ────────────────────────────────────────────────────────────
# Tell Docker that this container listens on port 8000.
# This is just documentation — actual port mapping happens at docker run time.
EXPOSE 8000

# ── 7. Start Command ──────────────────────────────────────────────────────────
# This runs when the container starts.
# --host 0.0.0.0 = listen on all network interfaces inside container
#                  without this, the API is only accessible inside the container
#                  and you can't reach it from your browser
# --port 8000    = serve on port 8000
# no --reload    = reload is for development only, not production
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]