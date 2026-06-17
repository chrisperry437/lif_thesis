# src/lif_thesis/streaming/sftp_watcher.py

from __future__ import annotations

import csv
import re
import time
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import paramiko


# =============================================================================
# CONFIGURATION
# =============================================================================

HOST = "192.168.1.103"
PORT = 22
USERNAME = "Rapid-E-user"
PASSWORD = "QEYKvnnw"

REMOTE_DIR = "/DATA/D_00001"

LOCAL_LIVE_DIR = Path("data/live_rapid_e")
LOCAL_ZIP_DIR = LOCAL_LIVE_DIR / "zips"
LOCAL_EXTRACT_DIR = LOCAL_LIVE_DIR / "extracted"
LOCAL_RAW_DIR = LOCAL_LIVE_DIR / "raw"
LOCAL_LOG_DIR = LOCAL_LIVE_DIR / "logs"

POLL_INTERVAL_SECONDS = 30

# Only ingest current live data by default.
MIN_YEAR = 2026

# During each poll, only process the newest N eligible files.
MAX_FILES_PER_POLL = 5

# Prevent downloading files that are still being written.
STABILITY_WAIT_SECONDS = 5

FILE_PATTERN = re.compile(r"^D_\d+_(\d{12})\.zip$")


# =============================================================================
# HELPERS
# =============================================================================

def ensure_dirs() -> None:
    for path in [
        LOCAL_LIVE_DIR,
        LOCAL_ZIP_DIR,
        LOCAL_EXTRACT_DIR,
        LOCAL_RAW_DIR,
        LOCAL_LOG_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def parse_timestamp(filename: str) -> Optional[datetime]:
    match = FILE_PATTERN.match(filename)

    if not match:
        return None

    return datetime.strptime(match.group(1), "%Y%m%d%H%M")


def is_eligible_live_file(filename: str) -> bool:
    timestamp = parse_timestamp(filename)

    if timestamp is None:
        return False

    return timestamp.year >= MIN_YEAR


def connect_sftp() -> tuple[paramiko.Transport, paramiko.SFTPClient]:
    transport = paramiko.Transport((HOST, PORT))
    transport.connect(username=USERNAME, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    return transport, sftp


def get_processed_files() -> set[str]:
    return {p.name for p in LOCAL_ZIP_DIR.glob("*.zip")}


def wait_until_remote_file_stable(
    sftp: paramiko.SFTPClient,
    remote_path: str,
) -> bool:
    size_1 = sftp.stat(remote_path).st_size
    time.sleep(STABILITY_WAIT_SECONDS)
    size_2 = sftp.stat(remote_path).st_size

    if size_1 != size_2:
        print(f"Skipping for now, still being written: {remote_path}")
        print(f"Size changed from {size_1} to {size_2}")
        return False

    return True


def download_file(
    sftp: paramiko.SFTPClient,
    remote_path: str,
    local_path: Path,
) -> bool:
    temp_path = local_path.with_suffix(local_path.suffix + ".partial")

    if local_path.exists():
        print(f"Already downloaded: {local_path}")
        return True

    if not wait_until_remote_file_stable(sftp, remote_path):
        return False

    print(f"Downloading: {remote_path}")

    try:
        sftp.get(remote_path, str(temp_path))
        temp_path.replace(local_path)
        print(f"Saved ZIP: {local_path}")
        return True

    except Exception as exc:
        print(f"Download failed: {remote_path}")
        print(exc)
        temp_path.unlink(missing_ok=True)
        return False


def extract_zip(zip_path: Path) -> Optional[Path]:
    extract_dir = LOCAL_EXTRACT_DIR / zip_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad_file = zf.testzip()

            if bad_file is not None:
                print(f"Bad file inside ZIP: {bad_file}")
                return None

            zf.extractall(extract_dir)

        return extract_dir

    except zipfile.BadZipFile:
        print(f"Bad or incomplete ZIP, deleting local copy: {zip_path}")
        zip_path.unlink(missing_ok=True)
        return None


def save_raw_files(
    extract_dir: Path,
    zip_name: str,
) -> list[Path]:
    raw_files = list(extract_dir.rglob("*.raw"))
    saved_files: list[Path] = []

    for raw_file in raw_files:
        target_name = f"{Path(zip_name).stem}__{raw_file.name}"
        target_path = LOCAL_RAW_DIR / target_name

        if not target_path.exists():
            target_path.write_bytes(raw_file.read_bytes())

        saved_files.append(target_path)

    return saved_files


def log_processed_file(
    zip_name: str,
    instrument_time: Optional[datetime],
    raw_files: list[Path],
) -> None:
    log_path = LOCAL_LOG_DIR / "processed_files.csv"
    file_exists = log_path.exists()

    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "zip_file",
                "instrument_time",
                "received_time_utc",
                "raw_file_count",
                "raw_files",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "zip_file": zip_name,
                "instrument_time": (
                    instrument_time.isoformat()
                    if instrument_time
                    else ""
                ),
                "received_time_utc": datetime.now(timezone.utc).isoformat(),
                "raw_file_count": len(raw_files),
                "raw_files": "|".join(str(p) for p in raw_files),
            }
        )


def process_zip(
    sftp: paramiko.SFTPClient,
    filename: str,
) -> bool:
    remote_path = f"{REMOTE_DIR}/{filename}"
    local_zip = LOCAL_ZIP_DIR / filename
    instrument_time = parse_timestamp(filename)

    downloaded = download_file(
        sftp=sftp,
        remote_path=remote_path,
        local_path=local_zip,
    )

    if not downloaded:
        return False

    extract_dir = extract_zip(local_zip)

    if extract_dir is None:
        return False

    raw_files = save_raw_files(
        extract_dir=extract_dir,
        zip_name=filename,
    )

    log_processed_file(
        zip_name=filename,
        instrument_time=instrument_time,
        raw_files=raw_files,
    )

    print("\nNew Rapid-E file processed")
    print(f"ZIP: {filename}")
    print(f"Instrument time: {instrument_time}")
    print(f"Received time UTC: {datetime.now(timezone.utc)}")
    print(f"Extracted to: {extract_dir}")
    print(f"RAW files saved: {len(raw_files)}")

    for raw_file in raw_files:
        size_mb = raw_file.stat().st_size / 1024 / 1024
        print(f"  - {raw_file} ({size_mb:.2f} MB)")

    return True


# =============================================================================
# MAIN LOOP
# =============================================================================

def run() -> None:
    ensure_dirs()

    processed_files = get_processed_files()

    print(f"Monitoring Rapid-E SFTP folder: {REMOTE_DIR}")
    print(f"Saving live data to: {LOCAL_LIVE_DIR}")
    print(f"Minimum year filter: {MIN_YEAR}")
    print(f"Loaded {len(processed_files)} previously downloaded ZIP files.")

    while True:
        transport = None
        sftp = None

        try:
            transport, sftp = connect_sftp()

            remote_files = sftp.listdir(REMOTE_DIR)

            eligible_files = sorted(
                [
                    f for f in remote_files
                    if (
                        FILE_PATTERN.match(f)
                        and is_eligible_live_file(f)
                        and f not in processed_files
                    )
                ],
                key=lambda f: parse_timestamp(f) or datetime.min,
            )

            new_files = eligible_files[-MAX_FILES_PER_POLL:]

            if not new_files:
                print(
                    f"No new eligible files found. Checked at "
                    f"{datetime.now(timezone.utc).isoformat()}"
                )

            for filename in new_files:
                success = process_zip(sftp, filename)

                if success:
                    processed_files.add(filename)

        except KeyboardInterrupt:
            print("\nStopping watcher.")
            break

        except Exception as exc:
            print(f"SFTP watcher error: {exc}")

        finally:
            if sftp is not None:
                sftp.close()
            if transport is not None:
                transport.close()

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run()