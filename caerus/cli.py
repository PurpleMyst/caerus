import sqlite3
from pathlib import Path
import typing as t

from .utils import FFMpeg, find_series, insert_if_not_exists
from .video_ops import (
    remove_segments,
    find_frame,
    matches_frame,
    nonblack,
    rfind_frame,
    video_length,
)
from tqdm import tqdm


class CLI:
    def __init__(self, database: str, ffmpeg_args: t.Dict[str, t.Any]) -> None:
        self.db = sqlite3.connect(database)
        self.db.executescript(
            Path(__file__).parent.joinpath("sql", "up.sql").read_text()
        )
        self.ffmpeg = FFMpeg(**ffmpeg_args)

    def mark(
        self,
        path: str,
        description: str,
        start: float,
        end: t.Optional[float],
    ) -> None:
        series = find_series(path)
        with self.db:
            series_id = insert_if_not_exists(self.db, "series", {"title": series})
            video_id = insert_if_not_exists(
                self.db,
                "videos",
                {"path": path, "series_id": series_id},
            )

            insert_if_not_exists(
                self.db,
                "markings",
                {
                    "description": description,
                    "video_id": video_id,
                    "start_timestamp": start,
                    "end_timestamp": end,
                },
            )

    def shave(self, path: str, output: str) -> None:
        series = find_series(path)

        rows: t.Iterable[t.Tuple[str, str, float, t.Optional[float]]] = self.db.execute(
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
            _, start_frame = find_frame(
                segment_path,
                nonblack(),
                offset=start_ts,
                desc="Searching first nonblack frame of segment",
            )

            if end_ts is None:
                start_pos, _ = rfind_frame(
                    path,
                    matches_frame(start_frame),
                    offset=-1,
                    desc="Searching backwards for start frame",
                )
                tqdm.write(f"Found {start_pos=}")
                end_pos = video_length(path)
            else:
                start_pos, _ = find_frame(
                    path,
                    matches_frame(start_frame),
                    desc="Searching for start frame",
                )
                tqdm.write(f"Found {start_pos=}")
                _, end_frame = rfind_frame(segment_path, nonblack(), offset=end_ts)
                end_pos, _ = rfind_frame(
                    path,
                    matches_frame(end_frame),
                    offset=start_pos + end_ts - start_ts,
                    desc="Searching for end frame",
                )
                tqdm.write(f"Found {end_pos=}")

            cutouts.append((start_pos, end_pos))

        remove_segments(path, output, cutouts, ffmpeg=self.ffmpeg)
