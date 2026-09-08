"""Shared test fixtures: a minimal from-scratch PNG encoder.

Deliberately independent of data/make_samples.py (which lives outside the
installed package) so tests don't depend on repo layout beyond `tests/`.
"""

from __future__ import annotations

import struct
import zlib


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def make_png_bytes(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    """A solid-color, uncompressed-filter RGB PNG -- enough to exercise parse_png."""
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = bytes(color) * width
    raw = bytearray()
    for _ in range(height):
        raw.append(0)
        raw.extend(row)
    idat = zlib.compress(bytes(raw), 9)
    return signature + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
