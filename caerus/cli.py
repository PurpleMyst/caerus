import sqlite3
import typing as t
from pathlib import Path
import logging
from functools import partial

import cv2
import structlog
import structlog.contextvars
from tqdm import tqdm

from .utils import FFMpeg, find_series, insert_unique
from .video_ops import (
    FoundFrame,
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
            series_id = insert_unique(self.db, "series", title=series)
            video_id = insert_unique(self.db, "videos", path=path, series_id=series_id)

            insert_unique(
                self.db,
                "markings",
                description=description,
                video_id=video_id,
                start_timestamp=start,
                end_timestamp=end,
            )

    def _query_markings(
        self, series: str
    ) -> t.List[t.Tuple[str, str, float, t.Optional[float]]]:
        return self.db.execute(
            """
            SELECT path, description, start_timestamp, end_timestamp
            FROM markings
            JOIN videos ON videos.id = video_id
            JOIN series ON series.title = ?""",
            (series,),
        ).fetchall()

    def _get_segment(
        self, path: str, start: float, end: t.Optional[float]
    ) -> t.Tuple[FoundFrame, t.Optional[FoundFrame]]:
        self.logger.info("looking for segment")

        segment_start = find_frame(
            path,
            nonblack,
            offset=start,
            desc="Searching first nonblack frame of segment",
        )
        self.logger.debug("found start of segment", ts=segment_start.ts)

        if end is None:
            segment_end = None
            self.logger.debug("found end of segment", ts=None)
        else:
            segment_end = rfind_frame(path, nonblack, offset=end)
            self.logger.debug("found end of segment", ts=segment_end.ts)

        return (segment_start, segment_end)

    def show_markings(self, path: str) -> None:
        series = find_series(path)
        markings = self._query_markings(series)
        for ref_path, desc, mark_start, mark_end in markings:
            structlog.contextvars.bind_contextvars(desc=desc)

            seg_start, seg_end = self._get_segment(ref_path, mark_start, mark_end)

            cv2.imshow(
                f"segment {desc!r} starting at {seg_start.ts:.3f} in {ref_path}",
                seg_start.frame,
            )
            if seg_end is not None:
                cv2.imshow(
                    f"segment {desc!r} ending at {seg_end.ts:.3f} in {ref_path}",
                    seg_end.frame,
                )
            cv2.waitKey()
            cv2.destroyAllWindows()

    def find_segments(self, path: str) -> t.List[t.Tuple[float, float]]:
        series = find_series(path)
        markings = self._query_markings(series)
        cutouts = []
        for ref_path, desc, mark_start, mark_end in markings:
            structlog.contextvars.bind_contextvars(desc=desc)

            seg_start, seg_end = self._get_segment(ref_path, mark_start, mark_end)

            start, _ = find_frame(
                path,
                partial(matches_frame, seg_start.frame),
                desc="Searching for start frame",
                h_offset=0.95 * seg_start.ts,
            )
            self.logger.info("found start in target", pos=start)

            if seg_end is None:
                end = video_length(path)
            else:
                end, _ = rfind_frame(
                    path,
                    partial(matches_frame, seg_end.frame),
                    offset=start + (seg_end.ts - seg_start.ts),
                    desc="Searching for end frame",
                )
            self.logger.info("found end in target", pos=end)

            cutouts.append((start, end))
        return cutouts

    def shave(self, path: str, output: str) -> None:
        cutouts = self.find_segments(path)
        remove_segments(path, output, cutouts, ffmpeg=self.ffmpeg)
