from pathlib import Path
import csv
import subprocess
import tempfile
import typing as t
import joblib
from time import perf_counter

import pydantic
from youtube_dl import YoutubeDL
from uuid import uuid1

from caerus.cli import CLI


class Timestamp(pydantic.BaseModel):
    description: str
    start: float
    end: t.Optional[float]


class Testcase(pydantic.BaseModel):
    url: str
    name: str
    timestamps: t.List[Timestamp]


VIDEOS_DIR = Path(__file__).parent.joinpath("videos")
MEMORY = joblib.Memory(VIDEOS_DIR)


@MEMORY.cache  # type: ignore
def get_video(tc: Testcase) -> str:
    unprocessed = str(VIDEOS_DIR / f"{tc.name}_unprocessed.%(ext)s")
    processed = str(VIDEOS_DIR / f"{tc.name}")

    with YoutubeDL(
        {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
            "outtmpl": unprocessed,
        }
    ) as ydl:
        ydl.download([tc.url])

    subprocess.run(
        (
            "filebot",
            "-rename",
            "--action",
            "move",
            unprocessed % {"ext": "mp4"},
            "--format",
            processed,
            "--conflict",
            "auto",
        ),
        check=True,
    )

    return processed + ".mp4"


def main() -> None:
    cli = CLI(":memory:", {})
    cases = pydantic.parse_file_as(
        t.List[Testcase], Path(__file__).parent.joinpath("data.json")
    )

    with tempfile.TemporaryDirectory() as dir, open("data.csv", "w", newline="") as f:
        w = csv.writer(f, dialect="excel")

        for case in cases:
            print(case)
            video = get_video(case)

            for ts in case.timestamps:
                start = perf_counter()
                cli.mark(video, ts.description, ts.start, ts.end)
                end = perf_counter()
                w.writerow((video, "mark", ts.description, start, end, end - start))

            start = perf_counter()
            cli.shave(video, str(Path(dir, f"{uuid1()}.mp4")))
            end = perf_counter()
            w.writerow((video, "shave", ts.description, start, end, end - start))


if __name__ == "__main__":
    main()
