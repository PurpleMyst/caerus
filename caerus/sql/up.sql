-- Video files that we have, associated to their series
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY,
    series_id INTEGER NOT NULL,
    path TEXT UNIQUE NOT NULL,

    FOREIGN KEY (series_id) REFERENCES series(id) ON UPDATE CASCADE ON DELETE CASCADE
);

-- Series, associated to their title
CREATE TABLE IF NOT EXISTS series (
    id INTEGER PRIMARY KEY,
    title TEXT UNIQUE NOT NULL
);

-- Segment references, utilized to find fluff in other videos of the same series
CREATE TABLE IF NOT EXISTS segment_references (
    id INTEGER PRIMARY KEY,
    video_id INTEGER NOT NULL,
    description TEXT UNIQUE,
    start REAL NOT NULL,
    end REAL, -- NULL represents "till the end of the video"

    FOREIGN KEY (video_id) REFERENCES videos(id) ON UPDATE CASCADE ON DELETE CASCADE
);

-- Segments found in particular files
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY,
    video_id INTEGER NOT NULL,
    reference_id INTEGER NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,

    FOREIGN KEY (reference_id) REFERENCES segment_references(id) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON UPDATE CASCADE ON DELETE CASCADE
);
