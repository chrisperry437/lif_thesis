"""
File watcher utilities for simulated or live Rapid-E data ingestion.

This module watches a directory for newly created files and yields them only
after they appear stable on disk. That prevents the pipeline from trying to
process a file while it is still being written.

Typical use:
    python -m lif_thesis.data.file_watcher data/realtime_mock
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable, Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


def is_file_stable(
    path: Path,
    stable_seconds: float = 2.0,
    check_interval: float = 0.5,
) -> bool:
    """
    Return True if a file's size remains unchanged for `stable_seconds`.

    Parameters
    ----------
    path:
        File path to check.
    stable_seconds:
        Number of seconds the file size must remain unchanged.
    check_interval:
        How often to check file size.

    Returns
    -------
    bool
        True if the file exists and appears stable.
    """
    if not path.exists() or not path.is_file():
        return False

    last_size = path.stat().st_size
    stable_for = 0.0

    while stable_for < stable_seconds:
        time.sleep(check_interval)

        if not path.exists() or not path.is_file():
            return False

        current_size = path.stat().st_size

        if current_size == last_size:
            stable_for += check_interval
        else:
            last_size = current_size
            stable_for = 0.0

    return True


def iter_existing_files(
    directory: Path | str,
    patterns: tuple[str, ...] = ("*",),
    recursive: bool = False,
) -> Iterator[Path]:
    """
    Yield existing files from a directory in sorted order.

    This is useful for replaying already-generated Rapid-E files as if they were
    arriving in real time.

    Parameters
    ----------
    directory:
        Directory to search.
    patterns:
        Glob patterns to include.
    recursive:
        Whether to search recursively.

    Yields
    ------
    Path
        Matching file paths.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    glob_method = directory.rglob if recursive else directory.glob

    seen: set[Path] = set()

    for pattern in patterns:
        for path in sorted(glob_method(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def watch_directory_polling(
    directory: Path | str,
    patterns: tuple[str, ...] = ("*",),
    poll_interval: float = 1.0,
    stable_seconds: float = 2.0,
    recursive: bool = False,
    include_existing: bool = False,
) -> Iterator[Path]:
    """
    Watch a directory for new files using simple polling.

    This avoids requiring the watchdog package and works reliably on Windows,
    Git Bash, Docker bind mounts, and network drives.

    Parameters
    ----------
    directory:
        Directory to watch.
    patterns:
        File patterns to watch, for example ("*.csv", "*.json", "*.parquet").
    poll_interval:
        Seconds between directory scans.
    stable_seconds:
        Seconds a file size must remain unchanged before yielding it.
    recursive:
        Whether to search subdirectories.
    include_existing:
        Whether to yield files already present when watching starts.

    Yields
    ------
    Path
        New stable files.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    processed: set[Path] = set()

    if include_existing:
        for path in iter_existing_files(directory, patterns, recursive):
            if is_file_stable(path, stable_seconds=stable_seconds):
                processed.add(path)
                yield path
    else:
        processed.update(iter_existing_files(directory, patterns, recursive))

    logger.info("Watching directory: %s", directory)

    glob_method = directory.rglob if recursive else directory.glob

    while True:
        candidates: list[Path] = []

        for pattern in patterns:
            candidates.extend(path for path in glob_method(pattern) if path.is_file())

        for path in sorted(set(candidates)):
            if path in processed:
                continue

            if is_file_stable(path, stable_seconds=stable_seconds):
                processed.add(path)
                logger.info("Detected new stable file: %s", path)
                yield path

        time.sleep(poll_interval)


def watch_directory(
    directory: Path | str,
    patterns: tuple[str, ...] = ("*",),
    poll_interval: float = 1.0,
    stable_seconds: float = 2.0,
    recursive: bool = False,
    include_existing: bool = False,
) -> Iterator[Path]:
    """
    Public wrapper for watching a directory.

    This function currently uses polling because it is simple, cross-platform,
    and reliable for thesis/demo deployment.
    """
    yield from watch_directory_polling(
        directory=directory,
        patterns=patterns,
        poll_interval=poll_interval,
        stable_seconds=stable_seconds,
        recursive=recursive,
        include_existing=include_existing,
    )


def process_new_files(
    directory: Path | str,
    callback: Callable[[Path], None],
    patterns: tuple[str, ...] = ("*",),
    poll_interval: float = 1.0,
    stable_seconds: float = 2.0,
    recursive: bool = False,
    include_existing: bool = False,
) -> None:
    """
    Watch a directory and call a function once for each new stable file.

    Parameters
    ----------
    directory:
        Directory to watch.
    callback:
        Function that accepts a Path and processes the file.
    patterns:
        File patterns to watch.
    poll_interval:
        Seconds between scans.
    stable_seconds:
        Seconds a file must remain unchanged before processing.
    recursive:
        Whether to search subdirectories.
    include_existing:
        Whether to process files already present at startup.
    """
    for path in watch_directory(
        directory=directory,
        patterns=patterns,
        poll_interval=poll_interval,
        stable_seconds=stable_seconds,
        recursive=recursive,
        include_existing=include_existing,
    ):
        try:
            callback(path)
        except Exception:
            logger.exception("Failed to process file: %s", path)


def _demo_callback(path: Path) -> None:
    """Simple demo callback used by the command-line interface."""
    print(f"New stable file detected: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a directory for new files.")
    parser.add_argument("directory", type=str, help="Directory to watch.")
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="File pattern to watch. Can be repeated. Example: --pattern '*.csv'",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between directory scans.",
    )
    parser.add_argument(
        "--stable-seconds",
        type=float,
        default=2.0,
        help="Seconds a file must remain unchanged before processing.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively.",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Process files that already exist when watcher starts.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()

    patterns = tuple(args.pattern) if args.pattern else ("*",)

    process_new_files(
        directory=args.directory,
        callback=_demo_callback,
        patterns=patterns,
        poll_interval=args.poll_interval,
        stable_seconds=args.stable_seconds,
        recursive=args.recursive,
        include_existing=args.include_existing,
    )


if __name__ == "__main__":
    main()