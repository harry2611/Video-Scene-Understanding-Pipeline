# Video Scene Understanding Pipeline

An end-to-end video scene understanding system with frame extraction, shot boundary detection, scene-level image classification, CLIP scene embeddings, FastAPI endpoints, Redis-backed asynchronous jobs, PostgreSQL metadata storage, and a React dashboard.

## What Is Implemented

### Level 1 - Basic

- MP4/local file and URL ingestion, including YouTube-compatible downloads through `yt-dlp`
- FFmpeg frame extraction at configurable FPS
- Scene boundary detection using both OpenCV histogram differences and optional PySceneDetect
- ResNet50 or EfficientNet-B0 frame classification through PyTorch/TorchVision
- Sliding-window temporal batching over extracted frame sequences
- Structured JSON metadata output with timestamps, labels, confidence scores, frame paths, and benchmark timings

### Level 2 - Intermediate

- CLIP embeddings per scene using Hugging Face Transformers
- FastAPI scene search endpoint for text queries
- Redis/RQ job queue for async video processing
- PostgreSQL persistence for job metadata, scene metadata, benchmark metrics, and embeddings
- React dashboard for submitting videos, viewing scenes, metrics, thumbnails, and semantic search results

### Level 3 - Advanced

- Temporal consistency filtering with blur detection, brightness thresholds, and per-frame quality flags
- Scene-level data quality scores with grades, recommended actions, and low-quality frame ratios
- PyTorch `DataParallel` support for multi-GPU classification throughput scaling
- Dashboard quality panels for real-time polling of metrics, scene thumbnails, quality scores, and review counts
- Benchmark comparison mode for single-device versus DataParallel throughput, with JSON and SVG chart output

## Quick Start

### 1. Install System Dependencies

Install FFmpeg first:

```bash
brew install ffmpeg
```

### 2. Python Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The first model-backed run may download TorchVision and CLIP weights.

### 3. Start PostgreSQL and Redis

```bash
docker compose up -d postgres redis
```

### 4. Run the API

```bash
uvicorn scene_pipeline.api.main:app --reload
```

### 5. Run the Worker

```bash
python -m scene_pipeline.worker.jobs
```

### 6. Run the Dashboard

```bash
cd frontend
npm install
npm run dev
```

Open the dashboard URL printed by Vite, usually `http://localhost:5173`.

## CLI Usage

Process a video directly and write metadata JSON:

```bash
python -m scene_pipeline.cli process ./video.mp4 --fps 1 --detector histogram --classifier resnet50 --output artifacts/metadata.json
```

Generate a lightweight benchmark report:

```bash
python scripts/benchmark.py ./video.mp4 --fps 1 --runs 1 --output reports/local_benchmark.json
```

Compare single-device and DataParallel classifier throughput:

```bash
python scripts/benchmark.py ./video.mp4 --fps 1 --runs 3 --compare-multi-gpu \
  --output reports/multi_gpu_benchmark.json \
  --chart-output reports/multi_gpu_benchmark.svg
```

## API Overview

- `POST /videos` submits a video source for async processing
- `POST /videos/sync` processes a video in the request for development/testing
- `GET /jobs/{job_id}` returns job status and metrics
- `GET /videos/{job_id}/metadata` returns full structured metadata
- `GET /search?query=street%20at%20night` returns CLIP text-to-scene matches
- `GET /metrics` returns aggregate processing metrics
- `GET /health` checks service readiness

## Configuration

Configuration is environment-variable driven. Common values:

```bash
DATABASE_URL=postgresql+psycopg://scene:scene@localhost:55432/scene_pipeline
REDIS_URL=redis://localhost:6379/0
PIPELINE_ARTIFACT_DIR=artifacts
PIPELINE_DEFAULT_FPS=1
PIPELINE_MODEL_NAME=resnet50
PIPELINE_SCENE_DETECTOR=histogram
PIPELINE_DEVICE=cpu
PIPELINE_ENABLE_MULTI_GPU=false
PIPELINE_ENABLE_QUALITY_SCORING=true
PIPELINE_BLUR_THRESHOLD=100
PIPELINE_BRIGHTNESS_MIN=35
PIPELINE_BRIGHTNESS_MAX=220
```

## Metadata Contract

The downstream JSON schema lives at [`schemas/video_metadata.schema.json`](schemas/video_metadata.schema.json). Each metadata document includes:

- Video identity and source
- Processing config
- Extracted frames with timestamps
- Scene time ranges and representative frames
- Scene labels and confidence scores
- Frame quality flags and scene quality scores
- Optional CLIP embeddings
- Temporal windows
- Per-stage latency and throughput measurements

## Notes

- PySceneDetect, Transformers, Torch, and TorchVision are optional at import time but required for their corresponding runtime features.
- CLIP vectors are stored as JSON arrays in PostgreSQL for portability. You can swap this to `pgvector` later without changing the public API shape.
- Redis queueing uses RQ. If Redis is unavailable, the synchronous endpoint remains useful for local development.
- DataParallel activates only when CUDA is selected and more than one GPU is visible to PyTorch. CPU and single-GPU runs still execute normally.
