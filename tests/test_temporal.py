from scene_pipeline.core.temporal import sliding_windows
from scene_pipeline.schemas import FrameMetadata


def _frames(count: int) -> list[FrameMetadata]:
    return [
        FrameMetadata(frame_index=index, timestamp=float(index), path=f"frame_{index}.jpg")
        for index in range(count)
    ]


def test_sliding_windows_include_tail_window() -> None:
    windows = sliding_windows(_frames(10), window_size=4, stride=3)

    assert [window.frame_indices for window in windows] == [
        [0, 1, 2, 3],
        [3, 4, 5, 6],
        [6, 7, 8, 9],
    ]


def test_sliding_windows_handles_short_sequence() -> None:
    windows = sliding_windows(_frames(3), window_size=8, stride=4)

    assert len(windows) == 1
    assert windows[0].frame_indices == [0, 1, 2]

