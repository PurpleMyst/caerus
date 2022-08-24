import typing as t
from itertools import islice
from math import ceil

import cv2
import imagehash
import numpy as np
import structlog
from colorama import Fore, Style
from PIL import Image
from tqdm import tqdm

from .utils import FFMpeg, Frame, log_parameters, releasing, format_time

BAR_FORMAT = (
    f"{Style.BRIGHT}{{desc}}{Style.RESET_ALL}: {Fore.GREEN}{{percentage:3.2f}}% "
    f"{Fore.BLUE}{{bar}}"
    f" {Fore.GREEN}{{n_fmt}}{Style.RESET_ALL}/{{total_fmt}}"
    f" [{Fore.GREEN}{{elapsed}} "
    f"{Style.BRIGHT}{Fore.RED}{{rate_fmt}}{Style.RESET_ALL}{{postfix}}]"
)


class FoundFrame(t.NamedTuple):
    ts: float
    frame: Frame


def video_length(cap: cv2.VideoCapture) -> float:
    return t.cast(float, cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))


def nonblack(frame: Frame, *, threshold: int = 32, percentage: float = 0.02) -> bool:
    """Check if FRAME is nonblack, i.e. not comprised of mostly black pixels.

    A frame is nonblack if PERCENTAGE pixels' luminance is greater than THRESHOLD."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nonblacks = np.count_nonzero(frame > threshold) / frame.size
    return nonblacks >= percentage


def nonwhite(frame: Frame, *, threshold: int = 223, percentage: float = 0.02) -> bool:
    """Check if FRAME is nonwhite, i.e. not comprised of mostly white pixels.

    A frame is nonwhite if PERCENTAGE pixels' luminance is lesser than THRESHOLD."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nonwhites = np.count_nonzero(frame < threshold) / frame.size
    return nonwhites >= percentage


def matches_frame(
    needle: Frame, candidate: Frame, *, percentage: float = 0.98, threshold: int = 32
) -> bool:
    if not nonblack(candidate):
        return False
    diff = cv2.absdiff(candidate, needle)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    matching = np.count_nonzero(np.less(diff, threshold))
    return matching / t.cast(float, diff.size) >= percentage


def frames(cap: cv2.VideoCapture, *, desc: str | None = None) -> t.Iterator[Frame]:
    with tqdm(
        total=cap.get(cv2.CAP_PROP_FRAME_COUNT) - cap.get(cv2.CAP_PROP_POS_FRAMES),
        bar_format=BAR_FORMAT,
        desc=desc,
    ) as pbar:
        while True:
            frame: Frame
            ok, frame = cap.read()
            if not ok:
                break
            pbar.update(1)
            yield frame


def unique_frames_hash(cap: cv2.VideoCapture, *, desc: str | None = None) -> set[imagehash.ImageHash]:
    return {imagehash.dhash(Image.fromarray(frame)) for frame in frames(cap, desc=desc)}


def intervals_from_hashes(
    hashes: set[imagehash.ImageHash], cap: cv2.VideoCapture, *, desc: str | None = None,
) -> list[tuple[float, float]]:
    result = []
    start = None
    gap = 0
    last_good = None

    # The amount of bad frames required to break an interval.
    break_at = cap.get(cv2.CAP_PROP_FPS)

    logger = structlog.get_logger().bind(video=desc, min_gap=break_at)
    for frame in frames(cap, desc=desc):
        if not (nonblack(frame) and nonwhite(frame)):
            continue

        h = imagehash.dhash(Image.fromarray(frame))
        if h in hashes:
            if gap != 0 and start is not None:
                logger.debug("interval.gap_clear", gap=gap)
            gap = 0

            last_good = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if start is None:
                start = last_good
                logger.debug(
                    "interval.start",
                    start=format_time(start),
                    start_hash=str(h),
                )
        elif start is not None:
            gap += 1

            if gap >= break_at:
                assert last_good is not None
                logger.info(
                    "interval.found",
                    start=format_time(start),
                    end=format_time(last_good),
                    end_hash=str(h),
                )
                result.append((start, last_good))
                start = last_good = None

    if start is not None:
        result.append((start, video_length(cap)))

    return result


@log_parameters(ignore=("predicate",))
def frames_satisfying(
    cap: cv2.VideoCapture,
    predicate: t.Callable[[Frame], bool],
    *,
    section: tuple[int, int | None] = (0, None),
    desc: str | None = None,
) -> t.Iterator[FoundFrame]:
    """Search for all frames satisfying PREDICATE in PATH.

    Parameters
    ----------
    path : str
        The path to the video in which to search.
    predicate : (Frame) => bool
        The predicate that must be satisfied.
    section : (int, int | None)
        The section of the video to search in.
    desc : str
        An optional description for the shown progress bar.
    """
    start, stop = section
    assert start >= 0 and (stop is None or stop >= 0)

    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    it = frames(cap, desc=desc)
    if stop is not None:
        stop_frames = (stop - start) / cap.get(cv2.CAP_PROP_FPS)
        it = islice(it, ceil(stop_frames))
    for frame in it:
        if predicate(frame):
            yield FoundFrame(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, frame)

@log_parameters()
def remove_segments(
    path: str,
    output: str,
    timestamps: list[tuple[float, float]],
    ffmpeg: FFMpeg = FFMpeg(),
) -> None:
    timestamps.sort()
    it = iter(timestamps)

    # Calculate the space in-between the segments to remove.
    first_start, prev_end = next(it)
    to_keep: list[tuple[float, float]] = []
    if first_start != 0:
        to_keep.append((0, first_start))

        for start, end in it:
            if prev_end != start:
                to_keep.append((prev_end, start))
            prev_end = end

    # If the last segment doesn't end with the video, make sure to keep everything after
    # the end of the last segment.
    with releasing(cv2.VideoCapture(path)) as cap:
        video_end = video_length(cap)
    if prev_end != video_end:
        to_keep.append((prev_end, video_end))

    # Calculate the ffmpeg concat filter to apply
    filters = []
    concat = []
    for i, (start, end) in enumerate(to_keep):
        # XXX is format=yuv420p needed/useful?
        filters.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[{i}v]")
        filters.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[{i}a]")
        concat.append(f"[{i}v][{i}a]")
    concat.append(f"concat=n={len(to_keep)}:v=1:a=1[outv][outa]")
    filters.append("".join(concat))

    ffmpeg(
        "-loglevel",
        "warning",
        "-stats",
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
        # Remove chapters as they may cause the output to have a wrong duration.
        "-map_chapters",
        "-1",
        output,
    )
