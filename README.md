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

Level 3 is intentionally not implemented yet. The codebase has extension points for quality scoring, multi-GPU benchmarking, and real-time metrics.

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
DATABASE_URL=postgresql+psycopg://scene:scene@localhost:5432/scene_pipeline
REDIS_URL=redis://localhost:6379/0
PIPELINE_ARTIFACT_DIR=artifacts
PIPELINE_DEFAULT_FPS=1
PIPELINE_MODEL_NAME=resnet50
PIPELINE_SCENE_DETECTOR=histogram
PIPELINE_DEVICE=auto
```

## Metadata Contract

The downstream JSON schema lives at [`schemas/video_metadata.schema.json`](schemas/video_metadata.schema.json). Each metadata document includes:

- Video identity and source
- Processing config
- Extracted frames with timestamps
- Scene time ranges and representative frames
- Scene labels and confidence scores
- Optional CLIP embeddings
- Temporal windows
- Per-stage latency and throughput measurements

## Notes

- PySceneDetect, Transformers, Torch, and TorchVision are optional at import time but required for their corresponding runtime features.
- CLIP vectors are stored as JSON arrays in PostgreSQL for portability. You can swap this to `pgvector` later without changing the public API shape.
- Redis queueing uses RQ. If Redis is unavailable, the synchronous endpoint remains useful for local development.

