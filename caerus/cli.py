import sqlite3
import typing as t
from pathlib import Path
import logging
from functools import partial

import structlog
import structlog.contextvars
from tqdm import tqdm

from .utils import FFMpeg, find_series, insert_if_not_exists
from .video_ops import (
    find_frame,
    matches_frame,
    nonblack,
    remove_segments,
    rfind_frame,
    video_length,
)


class TqdmWriteLogger:
    def msg(self, message: str) -> None:
        tqdm.write(message)

    log = debug = info = warn = warning = msg
    fatal = failure = err = error = critical = exception = msg


class CLI:
    def __init__(self, database: str, ffmpeg: FFMpeg) -> None:
        self.db = sqlite3.connect(database)
        self.db.executescript(
            Path(__file__).parent.joinpath("sql", "up.sql").read_text()
        )
        self.ffmpeg = ffmpeg

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.format_exc_info,
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                logging.DEBUG if __debug__ else logging.INFO
            ),
            context_class=dict,
            logger_factory=lambda *_: TqdmWriteLogger(),
            cache_logger_on_first_use=False,
        )
        self.logger = structlog.get_logger()

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

    def find_segments(self, path: str) -> t.List[t.Tuple[float, float]]:
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
        for segment_path, desc, mark_start_ts, mark_end_ts in rows:
            structlog.contextvars.bind_contextvars(desc=desc)
            self.logger.info("looking for segment")

            segment_start_ts, segment_start = find_frame(
                segment_path,
                nonblack,
                offset=mark_start_ts,
                desc="Searching first nonblack frame of segment",
            )
            self.logger.debug("found start of segment", ts=segment_start_ts)

            start_pos, _ = find_frame(
                path,
                partial(matches_frame, segment_start),
                desc="Searching for start frame",
                h_offset=0.95 * segment_start_ts,
            )
            self.logger.info("found start in target", start_pos=start_pos)

            if mark_end_ts is None:
                end_pos = video_length(path)
            else:
                segment_end_ts, segment_end = rfind_frame(
                    segment_path, nonblack, offset=mark_end_ts
                )

                self.logger.debug("found end of segment", ts=segment_end_ts)
                end_pos, _ = rfind_frame(
                    path,
                    partial(matches_frame, segment_end),
                    offset=start_pos + (segment_end_ts - segment_start_ts),
                    desc="Searching for end frame",
                )
            self.logger.info("found end in target", end_pos=end_pos)

            cutouts.append((start_pos, end_pos))
        return cutouts

    def shave(self, path: str, output: str) -> None:
        cutouts = self.find_segments(path)
        remove_segments(path, output, cutouts, ffmpeg=self.ffmpeg)
