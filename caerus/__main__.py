from __future__ import annotations

import json
import sqlite3
import subprocess
import typing as t
from pathlib import Path

import click

from tqdm import tqdm
from .video_ops import (
    find_frame,
    cutout,
    rfind_frame,
    nonblack,
    matches_frame,
    video_length,
)
from .utils import insert_if_not_exists, PathArg


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
    for segment_path, desc, start_ts, end_ts in rows:
        tqdm.write(f"Looking for {desc!r}")
        _, start_frame = find_frame(segment_path, nonblack(), offset=start_ts)
        start_pos, _ = find_frame(path, matches_frame(start_frame))
        tqdm.write(f"Found {start_pos=}")

        if end_ts is None:
            end_pos = video_length(path)
        else:
            _, end_frame = rfind_frame(segment_path, nonblack(), offset=end_ts)
            end_pos, _ = rfind_frame(
                path, matches_frame(end_frame), offset=start_pos + end_ts - start_ts
            )
            tqdm.write(f"Found {end_pos=}")

        cutouts.append((start_pos, end_pos))

    cutout(path, output, cutouts, preset=ctx.obj["preset"], crf=ctx.obj["crf"])


if __name__ == "__main__":
    cli()
