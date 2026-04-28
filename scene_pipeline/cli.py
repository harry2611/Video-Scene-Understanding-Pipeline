from __future__ import annotations

import argparse
import json
from pathlib import Path

from scene_pipeline.config import get_settings
from scene_pipeline.core.pipeline import PipelineOptions, VideoScenePipeline


def process_command(args: argparse.Namespace) -> None:
    settings = get_settings()
    pipeline = VideoScenePipeline(settings)
    options = PipelineOptions(
        fps=args.fps,
        scene_detector=args.detector,
        classifier=args.classifier,
        temporal_window_size=args.window_size,
        temporal_stride=args.stride,
        enable_clip=not args.disable_clip,
    )
    metadata = pipeline.process(args.source, options=options)
    output = Path(args.output) if args.output else None
    payload = json.dumps(metadata.model_dump(mode="json"), indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video scene understanding pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Process a video source")
    process.add_argument("source", help="Local MP4 path or remote/YouTube URL")
    process.add_argument("--fps", type=float, default=1.0)
    process.add_argument("--detector", choices=["histogram", "pyscenedetect"], default="histogram")
    process.add_argument("--classifier", choices=["resnet50", "efficientnet_b0"], default="resnet50")
    process.add_argument("--window-size", type=int, default=8)
    process.add_argument("--stride", type=int, default=4)
    process.add_argument("--disable-clip", action="store_true")
    process.add_argument("--output", help="Optional JSON output path")
    process.set_defaults(func=process_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

