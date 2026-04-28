from __future__ import annotations

import argparse
import json
from pathlib import Path

from scene_pipeline.config import get_settings
from scene_pipeline.core.pipeline import PipelineOptions, VideoScenePipeline


def run_benchmark(args: argparse.Namespace) -> dict:
    settings = get_settings()
    pipeline = VideoScenePipeline(settings)
    details = []

    for run_index in range(args.runs):
        options = PipelineOptions(
            fps=args.fps,
            scene_detector=args.detector,
            classifier=args.classifier,
            temporal_window_size=args.window_size,
            temporal_stride=args.stride,
            enable_clip=not args.disable_clip,
        )
        metadata = pipeline.process(args.source, job_id=f"benchmark-run-{run_index}", options=options)
        details.append(
            {
                "video_id": metadata.video_id,
                "frames_per_second": metadata.benchmark.frames_per_second,
                "total_latency_ms": metadata.benchmark.total_latency_ms,
                "total_frames": metadata.benchmark.total_frames,
                "scene_count": len(metadata.scenes),
                "stage_latencies": [
                    stage.model_dump(mode="json") for stage in metadata.benchmark.stage_latencies
                ],
            }
        )

    stage_names = sorted(
        {
            stage["name"]
            for detail in details
            for stage in detail["stage_latencies"]
        }
    )
    stage_latency_ms = {
        name: _average(
            [
                stage["latency_ms"]
                for detail in details
                for stage in detail["stage_latencies"]
                if stage["name"] == name
            ]
        )
        for name in stage_names
    }
    return {
        "source": args.source,
        "runs": args.runs,
        "averages": {
            "frames_per_second": _average([detail["frames_per_second"] for detail in details]),
            "total_latency_ms": _average([detail["total_latency_ms"] for detail in details]),
            "total_frames": _average([detail["total_frames"] for detail in details]),
            "scene_count": _average([detail["scene_count"] for detail in details]),
        },
        "stage_latency_ms": stage_latency_ms,
        "runs_detail": details,
    }


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the video scene pipeline")
    parser.add_argument("source", help="Local MP4 path or remote/YouTube URL")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--detector", choices=["histogram", "pyscenedetect"], default="histogram")
    parser.add_argument("--classifier", choices=["resnet50", "efficientnet_b0"], default="resnet50")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--disable-clip", action="store_true")
    parser.add_argument("--output", default="reports/local_benchmark.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    result = run_benchmark(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["averages"], indent=2))


if __name__ == "__main__":
    main()

