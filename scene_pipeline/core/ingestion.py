from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from scene_pipeline.schemas import FrameMetadata


def _is_remote_source(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"}


def resolve_video_source(source: str, job_dir: Path) -> Path:
    """Return a local video path, downloading remote sources through yt-dlp."""

    job_dir.mkdir(parents=True, exist_ok=True)
    local_source = Path(source).expanduser()
    if local_source.exists():
        target = job_dir / f"source{local_source.suffix or '.mp4'}"
        if local_source.resolve() != target.resolve():
            shutil.copy2(local_source, target)
        return target

    if not _is_remote_source(source):
        raise FileNotFoundError(f"Video source does not exist: {source}")

    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError("yt-dlp is required for remote video ingestion") from exc

    output_template = str(job_dir / "download.%(ext)s")
    options = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(source, download=True)
        downloaded = Path(ydl.prepare_filename(info))

    candidates = sorted(job_dir.glob("download*"))
    mp4_candidates = [path for path in candidates if path.suffix.lower() == ".mp4"]
    if mp4_candidates:
        return mp4_candidates[0]
    if downloaded.exists():
        return downloaded
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"yt-dlp did not create a downloadable artifact for {source}")


def probe_duration(video_path: Path) -> float | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    duration = payload.get("format", {}).get("duration")
    return float(duration) if duration is not None else None


def extract_frames(video_path: Path, output_dir: Path, fps: float) -> list[FrameMetadata]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = output_dir / "frame_%06d.jpg"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(frame_pattern),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)

    frames: list[FrameMetadata] = []
    for position, frame_path in enumerate(sorted(output_dir.glob("frame_*.jpg"))):
        width, height = _read_dimensions(frame_path)
        frames.append(
            FrameMetadata(
                frame_index=position,
                timestamp=position / fps,
                path=str(frame_path),
                width=width,
                height=height,
            )
        )
    return frames


def _read_dimensions(frame_path: Path) -> tuple[int | None, int | None]:
    try:
        import cv2
    except ImportError:
        return None, None
    image = cv2.imread(str(frame_path))
    if image is None:
        return None, None
    height, width = image.shape[:2]
    return width, height
