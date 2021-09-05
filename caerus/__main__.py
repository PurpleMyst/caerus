from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import typing as t
from contextlib import contextmanager
from itertools import repeat
from pathlib import Path

import click
import cv2
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

Frame = npt.NDArray[np.uint8]
PathArg = t.Union[str, bytes, os.PathLike[str], os.PathLike[bytes]]


@click.group()
@click.option("-d", "--database", type=click.Path(), default="database.db")
@click.option("--preset", type=str, default="medium")
@click.option("--crf", type=int, default="23")
@click.pass_context
def cli(ctx: click.Context, database: str, preset: str, crf: int) -> None:
    ctx.ensure_object(dict)
    db = ctx.obj["db"] = sqlite3.connect(str(database))
    ctx.obj["preset"] = preset
    ctx.obj["crf"] = crf

    db.executescript(Path(__file__).parent.joinpath("sql", "up.sql").read_text())


def find_series(path: PathArg) -> str:
    info = json.loads(
        subprocess.run(
            ("filebot", "-mediainfo", "--format", "{json}", path),
            check=True,
            capture_output=True,
        ).stdout.decode("ascii", "ignore")
    )
    series: str = info["seriesInfo"]["name"]
    return series


def insert_if_not_exists(
    db: sqlite3.Connection,
    table: str,
    values: t.Dict[str, t.Any],
) -> int:
    uniq_column, uniq_value = next(iter(values.items()))
    result = db.execute(
        f"SELECT id FROM {table} WHERE {uniq_column} = ?", (uniq_value,)
    ).fetchone()
    id: int
    if result is None:
        id = db.execute(
            f"INSERT INTO {table}({','.join(values.keys())}) "
            f"VALUES ({','.join(repeat('?', len(values)))})",
            tuple(values.values()),
        ).lastrowid
    else:
        [id] = result
    return id


@cli.command()
@click.argument("path", type=click.Path())
@click.option("-s", "--start", type=float, required=True)
@click.option("-e", "--end", type=float)
@click.option("-d", "--description", type=str, required=True)
@click.pass_context
def mark(
    ctx: click.Context,
    path: PathArg,
    start: float,
    end: t.Optional[float],
    description: str,
) -> None:
    db: sqlite3.Connection = ctx.obj["db"]

    with db:
        series_id = insert_if_not_exists(db, "series", {"title": find_series(path)})
        video_id = insert_if_not_exists(
            db,
            "videos",
            {"path": path, "series_id": series_id},
        )

        db.execute(
            "INSERT INTO "
            "markings(video_id, description, start_timestamp, end_timestamp) "
            "VALUES (?, ?, ?, ?)",
            (video_id, description, start, end),
        )


@contextmanager
def releasing(cap: cv2.VideoCapture) -> t.Iterator[cv2.VideoCapture]:
    try:
        yield cap
    finally:
        cap.release()


def find_frame(
    path: PathArg,
    predicate: t.Callable[[Frame], bool],
    *,
    offset: float = 0,
) -> t.Tuple[float, Frame]:
    with releasing(cv2.VideoCapture(path)) as cap:
        cap.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)
        pbar = tqdm(
            total=cap.get(cv2.CAP_PROP_FRAME_COUNT) - cap.get(cv2.CAP_PROP_POS_FRAMES)
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
) -> t.Tuple[float, Frame]:
    with releasing(cv2.VideoCapture(path)) as cap:
        cap.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        pbar = tqdm(total=pos)
        while True:
            pbar.update(1)
            frame: np.ndarray[t.Any, np.dtype[np.uint8]]
            ok, frame = cap.read()
            assert ok
            if predicate(frame):
                return (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, frame)
            pos -= 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)


def first_succeeding_nonblack_frame(
    path: PathArg,
    *,
    offset: float = 0,
    threshold: int = 25,
) -> t.Tuple[float, Frame]:
    return find_frame(
        path,
        lambda frame: np.any(frame > threshold),  # type: ignore
        offset=offset,
    )


def first_preceding_nonblack_frame(
    path: PathArg,
    *,
    offset: float = 0,
    threshold: int = 25,
) -> t.Tuple[float, Frame]:
    return rfind_frame(
        path,
        lambda frame: np.any(frame > threshold),  # type: ignore
        offset=offset,
    )


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

    end = get_end(path)
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


def get_end(path: PathArg) -> float:
    with releasing(cv2.VideoCapture(path)) as cap:
        return t.cast(
            float, cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        )


@cli.command()
@click.argument("path", type=click.Path())
@click.option("-o", "--output", type=click.Path(), default="out.mp4")
@click.pass_context
def shave(ctx: click.Context, path: PathArg, output: PathArg) -> None:
    """Remove found markings in a video file"""
    db: sqlite3.Connection = ctx.obj["db"]
    series = find_series(path)

    rows: t.Iterable[t.Tuple[str, str, float, t.Optional[float]]] = db.execute(
        """
        SELECT path, description, start_timestamp, end_timestamp
        FROM markings
        JOIN videos ON videos.id = video_id
        JOIN series ON series.title = ?""",
        (series,),
    ).fetchall()

    cutouts = []
    for segment_path, _, start_ts, end_ts in rows:
        _, start_frame = first_succeeding_nonblack_frame(segment_path, offset=start_ts)

        start_pos, _ = find_frame(
            path, lambda frame: np.array_equal(frame, start_frame)
        )

        if end_ts is None:
            end_pos = get_end(path)
        else:
            _, end_frame = first_preceding_nonblack_frame(segment_path, offset=end_ts)
            end_pos, _ = rfind_frame(
                path,
                lambda frame: np.array_equal(frame, end_frame),
                offset=start_pos + end_ts - start_ts,
            )

        cutouts.append((start_pos, end_pos))

    cutout(path, output, cutouts, preset=ctx.obj["preset"], crf=ctx.obj["crf"])


if __name__ == "__main__":
    cli()
