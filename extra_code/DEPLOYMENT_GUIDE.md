# Production Deployment Configuration

## 1. Requirements
```
fastapi==0.115.0
uvicorn[standard]==0.32.0
gunicorn==23.0.0
sentence-transformers==3.3.1
chromadb==0.5.20
torch==2.5.1
pydantic==2.10.3
```

## 2. Gunicorn Configuration (gunicorn_config.py)

```python
"""
Gunicorn configuration for production deployment.
Key settings for serving the embedding search API.
"""

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = 1  # CRITICAL: Only 1 worker to avoid loading model multiple times
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 10000  # Restart worker after 10k requests (memory leak prevention)
max_requests_jitter = 1000
timeout = 120  # 2 minutes for slow queries
keepalive = 5

# Logging
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "rag_embedding_search"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Preload app for faster worker spawning
preload_app = True  # Load model once in master, fork to workers

def on_starting(server):
    """Called just before the master process is initialized."""
    print("=" * 60)
    print("Starting RAG Embedding Search API")
    print(f"Workers: {workers}")
    print(f"Worker class: {worker_class}")
    print("=" * 60)

def worker_int(worker):
    """Called when a worker receives SIGINT or SIGQUIT."""
    print(f"Worker {worker.pid} received interrupt signal")

def worker_abort(worker):
    """Called when a worker receives SIGABRT."""
    print(f"Worker {worker.pid} aborted")
```

## 3. Environment Variables (.env)

```bash
# Model configuration
EMBEDDING_MODEL_NAME=infgrad/Jasper-Token-Compression-600M
CHROMADB_PATH=/mnt/e/data_files/embeddings

# Server configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1
LOG_LEVEL=info

# CUDA configuration
CUDA_VISIBLE_DEVICES=0  # Use GPU 0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 4. Systemd Service (for Linux)

```ini
[Unit]
Description=RAG Embedding Search API
After=network.target

[Service]
Type=notify
User=your_user
Group=your_group
WorkingDirectory=/path/to/your/app
Environment="PATH=/path/to/venv/bin"
EnvironmentFile=/path/to/.env
ExecStart=/path/to/venv/bin/gunicorn -c gunicorn_config.py fastapi_backend:app
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=30
PrivateTmp=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## 5. Docker Deployment

### Dockerfile
```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with gunicorn
CMD ["gunicorn", "-c", "gunicorn_config.py", "fastapi_backend:app"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - /path/to/chromadb:/data/chromadb:ro  # Read-only mount
    environment:
      - EMBEDDING_MODEL_NAME=infgrad/Jasper-Token-Compression-600M
      - CHROMADB_PATH=/data/chromadb
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

## 6. Running the API

### Development
```bash
# Direct uvicorn (single worker, auto-reload)
uvicorn fastapi_backend:app --reload --host 0.0.0.0 --port 8000
```

### Production (bare metal)
```bash
# With gunicorn (1 worker, production settings)
gunicorn -c gunicorn_config.py fastapi_backend:app
```

### Production (Docker)
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f rag-api

# Scale (NOT RECOMMENDED - model loading issues)
# docker-compose up -d --scale rag-api=1  # ALWAYS keep at 1
```

### Production (systemd)
```bash
# Enable and start service
sudo systemctl enable rag-api
sudo systemctl start rag-api

# Check status
sudo systemctl status rag-api

# View logs
sudo journalctl -u rag-api -f
```

## 7. Monitoring

### Prometheus metrics endpoint (optional add-on)
```python
# Add to fastapi_backend.py
from prometheus_fastapi_instrumentator import Instrumentator

@app.on_event("startup")
async def setup_metrics():
    Instrumentator().instrument(app).expose(app)
```

### Health check endpoint
```bash
# Check if service is healthy
curl http://localhost:8000/health

# Response
{
  "status": "healthy",
  "model_loaded": true,
  "collections_loaded": true,
  "collection_stats": {...},
  "system_info": {
    "cuda_available": true,
    "cuda_memory_allocated": 1.2
  }
}
```

## 8. Performance Tuning

### Memory Management
- **ChromaDB**: Runs on CPU RAM (~500MB for typical collections)
- **Model**: ~1.2GB VRAM on GPU (bfloat16)
- **Total per worker**: ~1.7GB combined

### Scaling Strategy
```
NEVER use multiple workers!

Why? Each worker loads the 600M model into VRAM.
- 1 worker: 1.2GB VRAM ✓
- 4 workers: 4.8GB VRAM ✗ (wasteful)

Instead:
1. Use 1 worker with async endpoints (FastAPI handles concurrency)
2. For horizontal scaling: Use multiple containers/servers behind load balancer
3. Each server runs 1 worker = efficient VRAM usage
```

### Load Balancer Setup (Nginx)
```nginx
upstream rag_api {
    # Multiple servers, each with 1 worker
    server 10.0.0.1:8000;
    server 10.0.0.2:8000;
    server 10.0.0.3:8000;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://rag_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 120s;
        proxy_read_timeout 120s;
    }
}
```

## 9. Testing

### Smoke test
```bash
# Health check
curl http://localhost:8000/health

# Simple search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "causal inference assumptions",
    "collection_num": 2,
    "k": 5
  }'

# List papers
curl http://localhost:8000/papers?collection_num=2
```

### Load test (locust)
```python
from locust import HttpUser, task, between

class RAGUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def search(self):
        self.client.post("/search", json={
            "query": "What is causal inference?",
            "collection_num": 2,
            "k": 10
        })
```

## 10. Troubleshooting

### Issue: Out of Memory
```bash
# Check GPU memory
nvidia-smi

# Solution: Reduce batch size or use CPU
# In embedding_search_fixed.py, set device="cpu"
```

### Issue: Slow startup
```bash
# Normal: Model loading takes 10-30 seconds
# Check logs for initialization messages

# Speed up: Use preload_app=True in gunicorn
```

### Issue: Connection refused
```bash
# Check if service is running
systemctl status rag-api

# Check logs
journalctl -u rag-api -n 100
```
