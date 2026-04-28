# Benchmarking Report

This report documents how to measure frames/sec throughput and latency per pipeline stage for the Level 1 and Level 2 implementation.

## Benchmark Command

```bash
python scripts/benchmark.py ./sample.mp4 --fps 1 --runs 3 --output reports/local_benchmark.json
```

## Metrics Captured

| Metric | Description |
| --- | --- |
| `frames_per_second` | End-to-end processed frames divided by total wall-clock runtime |
| `total_latency_ms` | Full pipeline runtime including ingestion, extraction, detection, classification, and optional CLIP |
| `stage_latencies` | Per-stage latency in milliseconds for ingestion, FFmpeg extraction, scene detection, temporal batching, classification, and CLIP embedding |
| `total_frames` | Number of extracted frames processed by classifiers and temporal batching |
| `scene_count` | Number of detected scene ranges |
| `average_quality_score` | Average scene data quality score from blur, brightness, and temporal consistency checks |

## Expected Observations

- FFmpeg extraction throughput scales primarily with source resolution, codec, and requested FPS.
- ResNet50 classification is usually the largest CPU-bound stage when running without GPU acceleration.
- CLIP embedding adds one image encoder pass per scene, so its cost scales with scene count instead of frame count.
- Data quality scoring adds one OpenCV pass per extracted frame and usually stays much cheaper than model inference.
- DataParallel improves classification throughput only when multiple CUDA GPUs are available and the batch size is large enough to feed them.
- Redis queue latency is intentionally excluded from direct pipeline throughput; API job wait time should be measured separately if needed.

## Single vs Multi-GPU Comparison

```bash
python scripts/benchmark.py ./sample.mp4 --fps 1 --runs 3 --compare-multi-gpu \
  --output reports/multi_gpu_benchmark.json \
  --chart-output reports/multi_gpu_benchmark.svg
```

The comparison mode runs two profiles:

| Profile | Classifier setting |
| --- | --- |
| `single_gpu` | `PipelineOptions(enable_multi_gpu=False)` |
| `data_parallel` | `PipelineOptions(enable_multi_gpu=True)` |

If only CPU or one CUDA GPU is available, the `data_parallel` profile still runs but reports no practical speedup because PyTorch does not wrap the model in `torch.nn.DataParallel`.

## Output Shape

The benchmark script writes a JSON object:

```json
{
  "source": "./sample.mp4",
  "runs": 3,
  "averages": {
    "frames_per_second": 2.74,
    "total_latency_ms": 5402.1,
    "scene_count": 8,
    "average_quality_score": 0.86
  },
  "stage_latency_ms": {
    "ingestion": 11.3,
    "frame_extraction": 891.5,
    "scene_detection": 104.2,
    "temporal_batching": 1.2,
    "frame_classification": 4210.6,
    "clip_embedding": 183.3
  },
  "runs_detail": []
}
```

The exact values depend on hardware, model cache state, video length, requested FPS, and whether CUDA is available.
