# Use the CPU Dockerfile as the default Docker build target for maximum compatibility.
FROM python:3.10-slim AS runtime

# Use /app as the application working directory inside the container.
WORKDIR /app

# Force Python logs to appear immediately in Docker logs.
ENV PYTHONUNBUFFERED=1
# Prevent pip from keeping large wheel caches inside image layers.
ENV PIP_NO_CACHE_DIR=1
# Expose local vendored repositories as importable Python packages.
ENV PYTHONPATH=/app:/app/GroundingDINO:/app/segment_anything

# Install Linux libraries needed by OpenCV and lightweight build tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency declarations first so Docker can cache dependency installation.
COPY requirements.txt /app/requirements.txt

# Upgrade packaging tools before installing Python dependencies.
RUN python -m pip install --upgrade pip setuptools wheel

# Install CPU-only PyTorch and torchvision so the default image works without NVIDIA drivers.
RUN python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Install the shared WebUI dependency set after PyTorch is already present.
RUN python -m pip install -r /app/requirements.txt

# Copy the WebUI source, backend, model repositories, and optional local weights.
COPY . /app

# Install Segment Anything in editable mode without reinstalling dependencies.
RUN python -m pip install --no-deps -e /app/segment_anything

# Document the FastAPI port exposed by the container.
EXPOSE 8000

# Start the FastAPI WebUI server in CPU-capable mode.
CMD ["uvicorn", "web_fastapi.app:app", "--host", "0.0.0.0", "--port", "8000"]
