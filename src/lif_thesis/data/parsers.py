from __future__ import annotations

import datetime as dt
import os
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


HEADER_FIXED_CODE = 0x22D5EBE7
FOOTER_FIXED_CODE = 0xF82F5BE4


@dataclass
class ParsedRawFile:
    particle_data: pd.DataFrame
    file_metadata: dict[str, Any]


def size_particle(scattering_image: list[int]) -> float | None:
    """Estimate particle size from scattering image."""
    x = np.asarray(scattering_image).sum()

    if x <= 0:
        return None

    return float(9.95e-01 * np.log(3.81e-05 * x) - 4.84e00)


def get_asymmetry(scattering_image: list[int]) -> float | None:
    """Estimate time asymmetry from scattering image."""
    if len(scattering_image) == 0:
        return None

    try:
        if len(scattering_image) < 3600:
            train_arr = np.asarray(scattering_image)
            train_ex = np.pad(train_arr, (0, 3600 - len(train_arr)), "constant").reshape(-1, 24)
        else:
            train_ex = np.asarray(scattering_image[0:3600]).reshape(-1, 24)

        extra_zeroes = np.zeros([40, 24])
        train_ex = np.vstack([extra_zeroes, train_ex, extra_zeroes])

        sumit = np.sum(train_ex, axis=1)
        if np.sum(sumit) == 0:
            return None

        x = np.linspace(0, 230, 230)
        p = int(np.dot(sumit, x) / np.sum(sumit))

        train_tosend = train_ex[p - 40 : p + 40].reshape(1, -1).tolist()[0]
        train_left = sorted(train_tosend[: 40 * 24])
        train_right = train_tosend[40 * 24 :]
        train_right_reverted = sorted(train_right[::-1])

        score = np.clip(abs(r2_score(train_left, train_right_reverted)), a_max=1, a_min=None)
        return float(1 - score)

    except Exception:
        return None


def _read_uint(file, n_bytes: int) -> int:
    return int.from_bytes(file.read(n_bytes), byteorder="big")


def _parse_timestamp(unix_seconds: int, unix_ms: int) -> str:
    timestamp = dt.datetime.fromtimestamp(unix_seconds)
    return f"{timestamp:%Y-%m-%d %H:%M:%S}.{unix_ms}"


def unzip_raw_file(zip_path: str | Path, keep_zip: bool = True) -> Path:
    """
    Unzip a Rapid-E zip file and return the extracted .raw path.
    """
    zip_path = Path(zip_path)
    output_dir = zip_path.parent

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        name = zip_ref.namelist()[0]
        zip_ref.extractall(output_dir)

    extracted_raw = output_dir / f"{Path(name).stem}.raw"

    expected_raw = zip_path.with_suffix(".raw")
    if extracted_raw != expected_raw and extracted_raw.exists():
        extracted_raw.rename(expected_raw)

    if not keep_zip:
        os.remove(zip_path)

    return expected_raw


def parse_raw_file(
    raw_path: str | Path,
    keep_thresholds: bool = False,
    extra_params: bool = True,
) -> ParsedRawFile:
    """
    Parse one Rapid-E .raw file.

    Output:
    - particle_data: one row per detected particle
    - file_metadata: file-level metadata and parsing diagnostics

    Based on the original project's conversion.py logic.
    """
    raw_path = Path(raw_path)

    if raw_path.suffix == ".zip":
        raw_path = unzip_raw_file(raw_path)

    rows: list[dict[str, Any]] = []
    nb_errors = 0
    serial = None
    version = None

    with raw_path.open("rb") as f:
        particle_index = 0

        while True:
            read_byte = f.read(4)

            if read_byte == b"":
                break

            header_fixed_code = int.from_bytes(read_byte, byteorder="big")

            if header_fixed_code != HEADER_FIXED_CODE:
                nb_errors += 1
                raise ValueError(
                    f"Invalid header in {raw_path.name}: "
                    f"{hex(header_fixed_code)} != {hex(HEADER_FIXED_CODE)}"
                )

            # General header
            version = _read_uint(f, 4)
            serial = _read_uint(f, 4)
            unix_time_seconds = _read_uint(f, 4)
            unix_time_ms = _read_uint(f, 4)
            timestamp = _parse_timestamp(unix_time_seconds, unix_time_ms)

            number_of_modules = _read_uint(f, 4)
            f.read(12)

            # Scattering header
            lt11_framelength = _read_uint(f, 4)
            scattering_emitter_id = _read_uint(f, 2)
            scattering_detection_id = _read_uint(f, 2)
            image_size = _read_uint(f, 4)
            f.read(20)

            lt03_framelength = None
            lt04_framelength = None

            if number_of_modules == 3:
                # Fluorescence header
                lt03_framelength = _read_uint(f, 4)
                fluo_emitter_id = _read_uint(f, 2)
                fluo_detection_id = _read_uint(f, 2)
                f.read(24)

                if lt03_framelength != 256:
                    nb_errors += 1

                # Lifetime header
                lt04_framelength = _read_uint(f, 4)
                lifetime_emitter_id = _read_uint(f, 2)
                lifetime_detection_id = _read_uint(f, 2)
                f.read(24)

                if lt04_framelength != 128:
                    nb_errors += 1

            # Scattering image: uint32 big-endian
            image_raw = f.read(image_size * 4)
            scattering_image = np.frombuffer(image_raw, dtype=np.dtype(">u4")).astype(int).tolist()
            crc = zlib.crc32(image_raw)

            # Scattering thresholds: uint32 big-endian
            threshold_raw = f.read((lt11_framelength - image_size) * 4)
            thresholds = np.frombuffer(threshold_raw, dtype=np.dtype(">u4")).astype(int).tolist()
            crc = zlib.crc32(threshold_raw, crc)

            spectrometer = None
            lifetime = None

            if number_of_modules == 3:
                # Spectrometer / fluorescence image: int32 big-endian
                fluo_raw = f.read(lt03_framelength * 4)
                spectrometer = np.frombuffer(fluo_raw, dtype=np.dtype(">i4")).astype(int).tolist()
                crc = zlib.crc32(fluo_raw, crc)

                # Lifetime image: int16 big-endian
                # Original parser reads lt04_framelength * 4 bytes and decodes int16,
                # producing 256 values when lt04_framelength = 128.
                life_raw = f.read(lt04_framelength * 4)
                lifetime = np.frombuffer(life_raw, dtype=np.dtype(">i2")).astype(int).tolist()
                crc = zlib.crc32(life_raw, crc)

            # Footer CRC
            footer_crc = _read_uint(f, 4)
            crc_ok = crc == footer_crc

            if not crc_ok:
                nb_errors += 1
                raise ValueError(
                    f"CRC mismatch in {raw_path.name}, particle {particle_index}: "
                    f"{hex(footer_crc)} != {hex(crc)}"
                )

            # Footer padding
            f.read(124)

            footer_fixed_code = _read_uint(f, 4)
            footer_ok = footer_fixed_code == FOOTER_FIXED_CODE

            if not footer_ok:
                nb_errors += 1
                raise ValueError(
                    f"Invalid footer in {raw_path.name}, particle {particle_index}: "
                    f"{hex(footer_fixed_code)} != {hex(FOOTER_FIXED_CODE)}"
                )

            row = {
                "raw_file": raw_path.name,
                "raw_path": str(raw_path),
                "particle_index": particle_index,
                "timestamp": timestamp,
                "serial": serial,
                "version": version,
                "number_of_modules": number_of_modules,
                "has_fluorescence": number_of_modules == 3,
                "lt11_framelength": lt11_framelength,
                "lt03_framelength": lt03_framelength,
                "lt04_framelength": lt04_framelength,
                "image_size": image_size,
                "scattering_emitter_id": scattering_emitter_id,
                "scattering_detection_id": scattering_detection_id,
                "scattering_image": scattering_image,
                "spectrometer": spectrometer,
                "lifetime": lifetime,
                "crc_ok": crc_ok,
                "footer_ok": footer_ok,
            }

            if keep_thresholds:
                row["scattering_thresholds"] = thresholds

            if extra_params:
                row["size"] = size_particle(scattering_image)
                row["time_asymmetry"] = get_asymmetry(scattering_image)

            rows.append(row)
            particle_index += 1

    particle_data = pd.DataFrame(rows)

    file_metadata = {
        "raw_file": raw_path.name,
        "raw_path": str(raw_path),
        "serial": serial,
        "version": version,
        "n_particles": len(particle_data),
        "n_errors": nb_errors,
        "file_size_bytes": raw_path.stat().st_size,
    }

    return ParsedRawFile(
        particle_data=particle_data,
        file_metadata=file_metadata,
    )