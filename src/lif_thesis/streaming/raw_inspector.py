# src/lif_thesis/streaming/raw_inspector.py

from __future__ import annotations

import re
import struct
from pathlib import Path
from datetime import datetime
from collections import Counter


# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_DIR = Path("data/live_rapid_e/raw")
OUTPUT_DIR = Path("data/live_rapid_e/inspection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "raw_inspection_report.csv"

TIMESTAMP_PATTERN = re.compile(r"_(\d{12})\.raw$")


# =============================================================================
# HELPERS
# =============================================================================

def parse_timestamp_from_raw_name(filename: str) -> datetime | None:
    match = TIMESTAMP_PATTERN.search(filename)

    if not match:
        return None

    return datetime.strptime(match.group(1), "%Y%m%d%H%M")


def read_bytes(path: Path, n_bytes: int = 256) -> bytes:
    with path.open("rb") as f:
        return f.read(n_bytes)


def hex_preview(data: bytes, max_bytes: int = 64) -> str:
    return data[:max_bytes].hex()


def ascii_preview(data: bytes, max_bytes: int = 64) -> str:
    preview = []

    for b in data[:max_bytes]:
        if 32 <= b <= 126:
            preview.append(chr(b))
        else:
            preview.append(".")

    return "".join(preview)


def unpack_preview(data: bytes) -> dict[str, list[int | float]]:
    """
    Try multiple interpretations of the first bytes.

    This does not assume we know the Rapid-E RAW format.
    It helps identify whether values look like:
    - unsigned integers
    - signed integers
    - floats
    - repeated binary records
    """

    preview: dict[str, list[int | float]] = {}

    sample = data[:64]

    formats = {
        "uint8": "B",
        "int8": "b",
        "uint16_be": ">H",
        "uint16_le": "<H",
        "int16_be": ">h",
        "int16_le": "<h",
        "uint32_be": ">I",
        "uint32_le": "<I",
        "int32_be": ">i",
        "int32_le": "<i",
        "float32_be": ">f",
        "float32_le": "<f",
    }

    for name, fmt in formats.items():
        size = struct.calcsize(fmt)
        values = []

        for i in range(0, len(sample) - size + 1, size):
            try:
                value = struct.unpack(fmt, sample[i:i + size])[0]
                values.append(value)
            except struct.error:
                continue

        preview[name] = values[:12]

    return preview


def byte_frequency(data: bytes) -> dict[int, int]:
    return dict(Counter(data))


def inspect_raw_file(path: Path) -> dict[str, object]:
    size_bytes = path.stat().st_size
    size_mb = size_bytes / 1024 / 1024

    first_256 = read_bytes(path, 256)

    timestamp = parse_timestamp_from_raw_name(path.name)

    unpacked = unpack_preview(first_256)

    return {
        "file": str(path),
        "filename": path.name,
        "timestamp": timestamp.isoformat() if timestamp else "",
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 3),
        "first_64_hex": hex_preview(first_256, 64),
        "first_64_ascii": ascii_preview(first_256, 64),
        "uint32_be_first_values": unpacked["uint32_be"],
        "uint32_le_first_values": unpacked["uint32_le"],
        "float32_be_first_values": unpacked["float32_be"],
        "float32_le_first_values": unpacked["float32_le"],
    }


def write_report(rows: list[dict[str, object]]) -> None:
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write(
            "filename,"
            "timestamp,"
            "size_bytes,"
            "size_mb,"
            "first_64_hex,"
            "first_64_ascii,"
            "uint32_be_first_values,"
            "uint32_le_first_values,"
            "float32_be_first_values,"
            "float32_le_first_values\n"
        )

        for row in rows:
            f.write(
                f'"{row["filename"]}",'
                f'"{row["timestamp"]}",'
                f'{row["size_bytes"]},'
                f'{row["size_mb"]},'
                f'"{row["first_64_hex"]}",'
                f'"{row["first_64_ascii"]}",'
                f'"{row["uint32_be_first_values"]}",'
                f'"{row["uint32_le_first_values"]}",'
                f'"{row["float32_be_first_values"]}",'
                f'"{row["float32_le_first_values"]}"\n'
            )


def print_summary(rows: list[dict[str, object]]) -> None:
    print("\nRAW inspection summary")
    print("======================")

    print(f"RAW directory: {RAW_DIR}")
    print(f"Files inspected: {len(rows)}")
    print(f"Report saved to: {REPORT_PATH}")

    if not rows:
        return

    sizes = [float(row["size_mb"]) for row in rows]

    print(f"\nSmallest file: {min(sizes):.3f} MB")
    print(f"Largest file: {max(sizes):.3f} MB")
    print(f"Average size: {sum(sizes) / len(sizes):.3f} MB")

    print("\nFirst 5 files:")
    for row in rows[:5]:
        print("--------------------------------")
        print(f"File: {row['filename']}")
        print(f"Timestamp: {row['timestamp']}")
        print(f"Size: {row['size_mb']} MB")
        print(f"HEX: {row['first_64_hex']}")
        print(f"ASCII: {row['first_64_ascii']}")
        print(f"uint32 BE: {row['uint32_be_first_values']}")
        print(f"uint32 LE: {row['uint32_le_first_values']}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"RAW directory does not exist: {RAW_DIR}"
        )

    raw_files = sorted(RAW_DIR.glob("*.raw"))

    rows = []

    for raw_file in raw_files:
        try:
            row = inspect_raw_file(raw_file)
            rows.append(row)
        except Exception as exc:
            print(f"Failed to inspect {raw_file}: {exc}")

    write_report(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()