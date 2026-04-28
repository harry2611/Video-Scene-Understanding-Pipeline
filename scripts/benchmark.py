from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from scene_pipeline.config import get_settings
from scene_pipeline.core.pipeline import PipelineOptions, VideoScenePipeline


def run_benchmark(args: argparse.Namespace) -> dict:
    if args.compare_multi_gpu:
        profiles = [
            _run_profile(args, profile="single_gpu", enable_multi_gpu=False),
            _run_profile(args, profile="data_parallel", enable_multi_gpu=True),
        ]
        return {
            "source": args.source,
            "runs": args.runs,
            "comparison": {
                "single_gpu_frames_per_second": profiles[0]["averages"]["frames_per_second"],
                "data_parallel_frames_per_second": profiles[1]["averages"]["frames_per_second"],
                "speedup": _speedup(
                    profiles[0]["averages"]["frames_per_second"],
                    profiles[1]["averages"]["frames_per_second"],
                ),
            },
            "profiles": profiles,
        }
    return _run_profile(args, profile="default", enable_multi_gpu=not args.disable_multi_gpu)


def _run_profile(args: argparse.Namespace, profile: str, enable_multi_gpu: bool) -> dict:
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
            enable_quality_scoring=not args.disable_quality_scoring,
            enable_multi_gpu=enable_multi_gpu,
        )
        metadata = pipeline.process(
            args.source,
            job_id=f"benchmark-{profile}-{run_index}",
            options=options,
        )
        details.append(
            {
                "video_id": metadata.video_id,
                "profile": profile,
                "multi_gpu_enabled": enable_multi_gpu,
                "frames_per_second": metadata.benchmark.frames_per_second,
                "total_latency_ms": metadata.benchmark.total_latency_ms,
                "total_frames": metadata.benchmark.total_frames,
                "scene_count": len(metadata.scenes),
                "average_quality_score": _average(
                    [
                        scene.quality.data_quality_score
                        for scene in metadata.scenes
                        if scene.quality is not None
                    ]
                ),
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
        "profile": profile,
        "multi_gpu_enabled": enable_multi_gpu,
        "source": args.source,
        "runs": args.runs,
        "averages": {
            "frames_per_second": _average([detail["frames_per_second"] for detail in details]),
            "total_latency_ms": _average([detail["total_latency_ms"] for detail in details]),
            "total_frames": _average([detail["total_frames"] for detail in details]),
            "scene_count": _average([detail["scene_count"] for detail in details]),
            "average_quality_score": _average(
                [detail["average_quality_score"] for detail in details]
            ),
        },
        "stage_latency_ms": stage_latency_ms,
        "runs_detail": details,
    }


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _speedup(baseline: float, candidate: float) -> float:
    return candidate / baseline if baseline > 0 else 0.0


def write_svg_chart(result: dict, output_path: Path) -> None:
    rows = _chart_rows(result)
    if not rows:
        return
    width = 760
    row_height = 54
    height = 96 + (len(rows) * row_height)
    label_width = 210
    bar_width = 420
    max_value = max(value for _, value, _ in rows) or 1.0

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 760 {height}" role="img">',
        "<style>"
        "text{font-family:Inter,Arial,sans-serif;fill:#172026}"
        ".muted{fill:#667684;font-size:12px}"
        ".label{font-size:14px;font-weight:700}"
        ".value{font-size:13px;font-weight:700}"
        ".axis{stroke:#dde3ea;stroke-width:1}"
        "</style>",
        f'<rect width="760" height="{height}" rx="8" fill="#ffffff"/>',
        '<text x="28" y="34" font-size="18" font-weight="800">Pipeline Benchmark</text>',
        '<text x="28" y="56" class="muted">Frames/sec and total latency comparison</text>',
        '<line x1="28" y1="76" x2="732" y2="76" class="axis"/>',
    ]
    for index, (label, value, unit) in enumerate(rows):
        y = 106 + (index * row_height)
        fill = "#1f7a5b" if "frames/sec" in label else "#4f8cff"
        length = max(4.0, (value / max_value) * bar_width)
        pieces.extend(
            [
                f'<text x="28" y="{y + 16}" class="label">{html.escape(label)}</text>',
                f'<rect x="{label_width}" y="{y}" width="{bar_width}" height="20" rx="5" fill="#edf1f5"/>',
                f'<rect x="{label_width}" y="{y}" width="{length:.2f}" height="20" rx="5" fill="{fill}"/>',
                f'<text x="{label_width + bar_width + 18}" y="{y + 16}" class="value">'
                f"{value:.2f} {html.escape(unit)}</text>",
            ]
        )
    pieces.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(pieces), encoding="utf-8")


def _chart_rows(result: dict) -> list[tuple[str, float, str]]:
    if "profiles" in result:
        rows: list[tuple[str, float, str]] = []
        for profile in result["profiles"]:
            name = str(profile["profile"]).replace("_", " ").title()
            rows.append((f"{name} frames/sec", profile["averages"]["frames_per_second"], "fps"))
            rows.append((f"{name} latency", profile["averages"]["total_latency_ms"], "ms"))
        return rows
    return [
        ("Default frames/sec", result["averages"]["frames_per_second"], "fps"),
        ("Default latency", result["averages"]["total_latency_ms"], "ms"),
    ]


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
    parser.add_argument("--disable-quality-scoring", action="store_true")
    parser.add_argument("--disable-multi-gpu", action="store_true")
    parser.add_argument("--compare-multi-gpu", action="store_true")
    parser.add_argument("--output", default="reports/local_benchmark.json")
    parser.add_argument("--chart-output", default="reports/local_benchmark.svg")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    result = run_benchmark(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_svg_chart(result, Path(args.chart_output))
    print(json.dumps(result.get("comparison", result.get("averages", {})), indent=2))


if __name__ == "__main__":
    main()
