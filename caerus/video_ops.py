import typing as t

import cv2
import numpy as np
import structlog
from tqdm import tqdm
from colorama import Fore, Style

from .utils import FFMpeg, Frame, releasing

BAR_FORMAT = (
    f"{Style.BRIGHT}{{desc}}{Style.RESET_ALL}: {Fore.GREEN}{{percentage:3.2f}}% "
    f"{Fore.BLUE}{{bar}}"
    f" {Fore.GREEN}{{n_fmt}}{Style.RESET_ALL}/{{total:.0f}}"
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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nonblacks = np.count_nonzero(frame > threshold) / frame.size
    return nonblacks >= percentage


def matches_frame(
    needle: Frame, candidate: Frame, *, percentage: float = 0.98, threshold: int = 32
) -> bool:
    if not nonblack(candidate):
        return False
    diff = cv2.absdiff(candidate, needle)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    matching = np.count_nonzero(np.less(diff, threshold))
    return matching / diff.size >= percentage  # type: ignore


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
    structlog.get_logger().debug(
        "find_frame", path=path, offset=offset, h_offset=h_offset
    )
    if h_offset is not None:
        try:
            return find_frame(
                path, predicate, offset=h_offset, desc=f"{desc} (heuristically)"
            )
        except LookupError:
            pass

    with releasing(cv2.VideoCapture(path)) as cap:
        cap.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)
        pbar = tqdm(
            total=cap.get(cv2.CAP_PROP_FRAME_COUNT) - cap.get(cv2.CAP_PROP_POS_FRAMES),
            desc=desc,
            bar_format=BAR_FORMAT,
        )
        while True:
            pbar.update(1)
            frame: np.ndarray[t.Any, np.dtype[np.uint8]]
            ok, frame = cap.read()
            if not ok:
                raise LookupError
            if predicate(frame):
                return FoundFrame(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, frame)


def rfind_frame(
    path: str,
    predicate: t.Callable[[Frame], bool],
    *,
    offset: float = 0,
    desc: t.Optional[str] = None,
) -> FoundFrame:
    structlog.get_logger().debug("rfind_frame", path=path, offset=offset)
    with releasing(cv2.VideoCapture(path)) as cap:
        if offset < 0:
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)

        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if offset < 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos + offset)
        pbar = tqdm(total=pos, desc=desc, bar_format=BAR_FORMAT)
        while True:
            assert pos >= 0
            pbar.update(1)
            frame: Frame
            ok, frame = cap.read()
            if not ok:
                raise LookupError
            if predicate(frame):
                return FoundFrame(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, frame)
            pos -= 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)


def remove_segments(
    path: str,
    output: str,
    timestamps: t.List[t.Tuple[float, float]],
    ffmpeg: FFMpeg = FFMpeg(),
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
