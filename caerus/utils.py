import json
import sqlite3
import typing as t
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import repeat
import subprocess

import cv2
import numpy as np
import numpy.typing as npt

Frame = npt.NDArray[np.uint8]


@contextmanager
def releasing(cap: cv2.VideoCapture) -> t.Iterator[cv2.VideoCapture]:
    try:
        yield cap
    finally:
        cap.release()


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


@dataclass
class FFMpeg:
    options: t.Dict[str, t.Any] = field(default_factory=dict)

    def __call__(
        self, *args: t.Any, **kwargs: t.Any
    ) -> subprocess.CompletedProcess[str]:
        kwargs.setdefault("check", True)
        cmd = ["ffmpeg"]
        *head, tail = args
        cmd.extend(map(str, head))
        for k, v in self.options.items():
            cmd.append("-" + k)
            cmd.append(str(v))
        cmd.append(str(tail))
        return subprocess.run(cmd, **kwargs)


def find_series(path: str) -> str:
    info = json.loads(
        subprocess.run(
            ("filebot", "-mediainfo", "--format", "{json}", path),
            check=True,
            capture_output=True,
        ).stdout.decode("ascii", "ignore")
    )
    series: str = info["seriesInfo"]["name"]
    return series
