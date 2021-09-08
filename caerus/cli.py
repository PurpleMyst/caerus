import logging
import sqlite3
import sys
import typing as t
from functools import partial
from pathlib import Path

import cv2
import orjson
import structlog
import structlog.contextvars
import structlog_overtime
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
    def __init__(self, *_: t.Any) -> None:
        self._write = partial(tqdm.write, file=sys.stderr)

    def msg(self, message: str) -> None:
        self._write(message)

    log = debug = info = warn = warning = msg
    fatal = failure = err = error = critical = exception = msg


class CLI:
    def __init__(self, database: str) -> None:
        self.db = sqlite3.connect(database)
        self.db.executescript((Path(__file__).parent / "sql" / "up.sql").read_text())

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.format_exc_info,
                structlog.processors.TimeStamper(fmt="ISO"),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                logging.DEBUG if __debug__ else logging.INFO
            ),
            context_class=dict,
            logger_factory=structlog_overtime.TeeLoggerFactory(
                structlog_overtime.TeeOutput(
                    processors=[structlog.dev.ConsoleRenderer()],
                    logger_factory=TqdmWriteLogger,
                ),
                structlog_overtime.TeeOutput(
                    processors=[
                        structlog.processors.JSONRenderer(serializer=orjson.dumps)
                    ],
                    logger_factory=structlog.BytesLoggerFactory(
                        open("logs.jsonl", "ab")
                    ),
                ),
            ),
            cache_logger_on_first_use=True,
        )
        self.logger = structlog.get_logger().bind()

    def add_reference(
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

            self.db.execute(
                "INSERT INTO segment_references(description, video_id, start, end) "
                "VALUES (?, ?, ?, ?)",
                (description, video_id, start, end),
            )

    def remove_reference(
        self,
        description: str,
    ) -> None:
        with self.db:
            self.db.execute(
                "DELETE FROM segment_references WHERE description=?", (description,)
            )

    def _query_segment_references_for_series(
        self, series: str
    ) -> t.List[t.Tuple[int, str, str, float, t.Optional[float]]]:
        return self.db.execute(
            "SELECT segment_references.id, path, description, start, end"
            " FROM segment_references"
            " JOIN videos ON videos.id = video_id"
            " JOIN series ON series.title = ?",
            (series,),
        ).fetchall()

    def _query_unfound_segment_references(
        self, video_id: int, series_id: int
    ) -> t.List[t.Tuple[int, str, str, float, t.Optional[float]]]:
        return self.db.execute(
            "SELECT segment_references.id, path, description, start, end"
            " FROM segment_references"
            " JOIN videos ON videos.id = video_id"
            " JOIN series ON series.id = ?"
            " WHERE segment_references.id NOT IN ("
            "  SELECT reference_id"
            "  FROM segments"
            "  WHERE video_id = ?"
            ")",
            (series_id, video_id),
        ).fetchall()

    def _query_segment_references_for_path(
        self, path: str
    ) -> t.List[t.Tuple[int, str, str, float, t.Optional[float]]]:
        return self.db.execute(
            "SELECT segment_references.id, description, start, end"
            " FROM segment_references"
            " JOIN videos ON videos.path = ?",
            (path,),
        ).fetchall()

    def _get_segment_ref(
        self, path: str, start: float, end: t.Optional[float]
    ) -> t.Tuple[FoundFrame, t.Optional[FoundFrame]]:
        logger = self.logger.bind(path=path)
        logger.debug("_get_segment_ref", start=start, end=end)

        segment_start = find_frame(
            path,
            nonblack,
            offset=start,
            desc="Searching for the first nonblack frame",
        )
        logger.debug("found start of segment reference", ts=segment_start.ts)

        if end is None:
            segment_end = None
            logger.debug("found end of segment reference", ts=None)
        else:
            segment_end = rfind_frame(path, nonblack, offset=end)
            logger.debug("found end of segment reference", ts=segment_end.ts)

        return (segment_start, segment_end)

    def show_references(self, path: str, all_in_series: bool) -> None:
        if all_in_series:
            series = find_series(path)
            references = self._query_segment_references_for_series(series)
        else:
            references = self._query_segment_references_for_path(path)
        for id_, path, desc, start, end in references:
            structlog.contextvars.bind_contextvars(desc=desc)

            seg_start, seg_end = self._get_segment_ref(path, start, end)

            cv2.imshow(
                f"segment reference {id_} {desc!r} "
                f"starting at {seg_start.ts:.3f} in {path}",
                seg_start.frame,
            )
            if seg_end is not None:
                cv2.imshow(
                    f"segment reference {id_} {desc!r} "
                    f"ending at {seg_end.ts:.3f} in {path}",
                    seg_end.frame,
                )
            cv2.waitKey()
            cv2.destroyAllWindows()

    def find_segments(self, path: str) -> None:
        structlog.contextvars.bind_contextvars(target=path)
        with self.db:
            series_id = insert_unique(self.db, "series", title=find_series(path))
            video_id = insert_unique(self.db, "videos", path=path, series_id=series_id)
        references = self._query_unfound_segment_references(video_id, series_id)

        self.logger.debug("unfound_references", references=references)
        segments = []
        for reference_id, ref_path, desc, ref_start, ref_end in references:
            structlog.contextvars.bind_contextvars(desc=desc)

            seg_start, seg_end = self._get_segment_ref(ref_path, ref_start, ref_end)

            try:
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
            except LookupError as e:
                self.logger.warn("segment not found", error=e)
                continue

            segments.append((reference_id, start, end))
        structlog.contextvars.unbind_contextvars("desc")

        with self.db:
            for reference_id, start, end in segments:
                self.db.execute(
                    "INSERT INTO segments(video_id, reference_id, start, end)"
                    "VALUES (?, ?, ?, ?)",
                    (video_id, reference_id, start, end),
                )

    def found_segments(self, path: str) -> t.List[t.Tuple[float, float]]:
        return self.db.execute(
            "SELECT start, end"
            " FROM segments"
            " JOIN videos ON video_id = segments.id AND path = ?",
            (path,),
        ).fetchall()

    def shave(self, path: str, output: str, ffmpeg: FFMpeg) -> None:
        self.find_segments(path)
        remove_segments(path, output, self.found_segments(path), ffmpeg)
