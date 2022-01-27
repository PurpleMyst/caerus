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


def video_length(path: str) -> float:
    with releasing(cv2.VideoCapture(path)) as cap:
        return t.cast(
            float, cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        )


def nonblack(frame: Frame, *, threshold: int = 32, percentage: float = 0.02) -> bool:
    """Check if FRAME is nonblack, i.e. not comprised of mostly black pixels.

    A frame is nonblack if PERCENTAGE% pixels' luminance is greater than THRESHOLD."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nonblacks = np.count_nonzero(frame > threshold) / frame.size
    return nonblacks >= percentage


def nonwhite(frame: Frame, *, threshold: int = 223, percentage: float = 0.02) -> bool:
    """Check if FRAME is nonwhite, i.e. not comprised of mostly white pixels.

    A frame is nonwhite if PERCENTAGE% pixels' luminance is lesser than THRESHOLD."""
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
    return matching / diff.size >= percentage  # type: ignore


def frames(cap: cv2.VideoCapture, *, desc: t.Optional[str] = None) -> t.Iterator[Frame]:
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


def unique_frames_hash(video: str) -> t.Set[imagehash.ImageHash]:
    with releasing(cv2.VideoCapture(video)) as cap:
        return {
            imagehash.dhash(Image.fromarray(frame)) for frame in frames(cap, desc=video)
        }


def intervals_from_hashes(
    hashes: t.Set[imagehash.ImageHash], video: str
) -> t.List[t.Tuple[float, float]]:
    result = []
    start = None
    gap = 0
    last_good = None

    with releasing(cv2.VideoCapture(video)) as cap:
        min_gap = cap.get(cv2.CAP_PROP_FPS)

        logger = structlog.get_logger().bind(video=video, min_gap=min_gap)
        for frame in frames(cap, desc=video):
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

                if gap >= min_gap:
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
        result.append((start, video_length(video)))

    return result


@log_parameters(ignore=("predicate",))
def find_frame(
    path: str,
    predicate: t.Callable[[Frame], bool],
    *,
    offset: float = 0,
    h_offset: t.Optional[float] = None,
    desc: t.Optional[str] = None,
) -> FoundFrame:
    """Search for a frame satisfying PREDICATE in PATH.

    Parameters
    ----------
    path : str
        The path to the video in which to search
    predicate: (Frame) => bool
        The predicate that must be satisfied
    offset : float
        Where to seek the video to before searching
    h_offset : float
        An "heuristic" offset. This offset is where you guess the frame will be found.
        The search will start from here and, shall the search from that offset fail,
        it'll be retried starting from OFFSET.
    desc : str
        An optional description for the shown progress bar
    """
    logger = structlog.get_logger()
    if h_offset is not None:
        try:
            return find_frame(
                path, predicate, offset=h_offset, desc=f"{desc} (heuristically)"
            )
        except LookupError:
            pass

    with releasing(cv2.VideoCapture(path)) as cap:
        cap.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)
        it = frames(cap, desc=desc)
        if h_offset is not None:
            end = (h_offset - offset) / cap.get(cv2.CAP_PROP_FPS)
            logger.debug("find_frame stopping at", end=end)
            it = islice(it, ceil(end))
        for frame in it:
            if predicate(frame):
                return FoundFrame(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, frame)
    raise LookupError


@log_parameters(ignore=("predicate",))
def rfind_frame(
    path: str,
    predicate: t.Callable[[Frame], bool],
    *,
    offset: float = 0,
    desc: t.Optional[str] = None,
) -> FoundFrame:
    with releasing(cv2.VideoCapture(path)) as cap:
        if offset < 0:
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)

        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if offset < 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos + offset)
        with tqdm(desc=desc, bar_format=BAR_FORMAT, total=pos + 1) as pbar:
            while True:
                if pos < 0:
                    raise LookupError
                pbar.update(1)
                frame: Frame
                ok, frame = cap.read()
                if not ok:
                    raise LookupError
                if predicate(frame):
                    return FoundFrame(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, frame)
                pos -= 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)


@log_parameters()
def remove_segments(
    path: str,
    output: str,
    timestamps: t.List[t.Tuple[float, float]],
    ffmpeg: FFMpeg = FFMpeg(),
) -> None:
    timestamps.sort()
    it = iter(timestamps)

    # Calculate the space in-between the segments to remove.
    first_start, prev_end = next(it)
    to_keep: t.List[t.Tuple[float, float]] = []
    if first_start != 0:
        to_keep.append((0, first_start))

        for start, end in it:
            if prev_end != start:
                to_keep.append((prev_end, start))
            prev_end = end

    # If the last segment doesn't end with the video, make sure to keep everything after
    # the end of the last segment.
    end = video_length(path)
    if prev_end != end:
        to_keep.append((prev_end, end))

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
