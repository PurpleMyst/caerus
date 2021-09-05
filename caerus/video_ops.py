import subprocess
import typing as t

import cv2
import numpy as np
from tqdm import tqdm

from .utils import Frame, PathArg, releasing


def video_length(path: PathArg) -> float:
    with releasing(cv2.VideoCapture(path)) as cap:
        return t.cast(
            float, cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        )


def nonblack(
    threshold: int = 32, percentage: float = 0.02
) -> t.Callable[[Frame], bool]:
    def predicate(frame: Frame) -> bool:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nonblacks = np.count_nonzero(frame > threshold) / frame.size
        return nonblacks >= percentage

    return predicate


def matches_frame(
    needle: Frame, *, percentage: float = 0.98, threshold: int = 32
) -> t.Callable[[Frame], bool]:
    """Return a predicate for find_frame that searches for matches to the given frame

    Parameters
    ----------
    percentage : float
        How much of the image should match?
    threshold : int
        The images are compared by being subtracted from one another and then converted
        to luminance values. A pixel is considered as being the same if its "difference
        luminance" is less than or equal to the threshold parameter.
    """

    is_nonblack = nonblack()

    def predicate(frame: Frame) -> bool:
        if not is_nonblack(frame):
            return False
        diff = cv2.absdiff(frame, needle)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        matching = np.count_nonzero(np.less(diff, threshold))
        return matching / diff.size >= percentage  # type: ignore

    return predicate


def find_frame(
    path: PathArg,
    predicate: t.Callable[[Frame], bool],
    *,
    offset: float = 0,
    desc: t.Optional[str] = None,
) -> t.Tuple[float, Frame]:
    with releasing(cv2.VideoCapture(path)) as cap:
        cap.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)
        pbar = tqdm(
            total=cap.get(cv2.CAP_PROP_FRAME_COUNT) - cap.get(cv2.CAP_PROP_POS_FRAMES),
            desc=desc,
        )
        while True:
            pbar.update(1)
            frame: np.ndarray[t.Any, np.dtype[np.uint8]]
            ok, frame = cap.read()
            assert ok
            if predicate(frame):
                return (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, frame)


def rfind_frame(
    path: PathArg,
    predicate: t.Callable[[Frame], bool],
    *,
    offset: float = 0,
    desc: t.Optional[str] = None,
) -> t.Tuple[float, Frame]:
    with releasing(cv2.VideoCapture(path)) as cap:
        if offset < 0:
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)

        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if offset < 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos + offset)
        pbar = tqdm(total=pos, desc=desc)
        while True:
            assert pos >= 0
            pbar.update(1)
            frame: np.ndarray[t.Any, np.dtype[np.uint8]]
            ok, frame = cap.read()
            assert ok
            if predicate(frame):
                return (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, frame)
            pos -= 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)


def cutout(
    path: PathArg,
    output: PathArg,
    timestamps: t.List[t.Tuple[float, float]],
    *,
    preset: str,
    crf: int,
) -> None:
    timestamps.sort()
    it = iter(timestamps)
    first_start, prev_end = next(it)

    to_keep: t.List[t.Tuple[float, float]] = []
    if first_start != 0:
        to_keep.append((0, first_start))

        for start, end in it:
            if prev_end != start:
                to_keep.append((prev_end, start))
            prev_end = end

    end = video_length(path)
    if prev_end != end:
        to_keep.append((prev_end, end))

    filters = []
    concat = []

    for i, (start, end) in enumerate(to_keep):
        # XXX is format=yuv420p needed/useful?
        filters.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[{i}v]")
        filters.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[{i}a]")
        concat.append(f"[{i}v][{i}a]")

    concat.append(f"concat=n={len(to_keep)}:v=1:a=1[outv][outa]")
    filters.append("".join(concat))

    # FIXME it looks like maybe the output video has the wrong length metadata but it
    # plays correctly?
    subprocess.run(
        (
            "ffmpeg",
            "-loglevel",
            "warning",
            "-i",
            path,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outv]",
            "-map",
            "[outa]",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            output,
        ),
        check=True,
    )
