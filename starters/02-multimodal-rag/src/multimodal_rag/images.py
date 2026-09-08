"""PNG introspection (stdlib-only) and Anthropic image content-block helpers.

The offline StubClient captions images by actually decoding their pixels
-- width, height, and two real, deterministically-computed color
statistics -- so `--offline` produces a genuinely honest demo, not a
canned string. That means this decoder implements real PNG un-filtering
(_defilter, covering all five PNG filter types), not just the flat
filter-0 rows this starter's own data/make_samples.py happens to write.
"""

from __future__ import annotations

import base64
import struct
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SUPPORTED_COLOR_TYPES = {2: 3, 6: 4}  # PNG color type -> channels (RGB / RGBA)


@dataclass(frozen=True)
class PngInfo:
    width: int
    height: int
    color_type: int
    channels: int
    pixels: bytes | None  # None when pixel decoding isn't supported for this PNG


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _defilter(raw: bytes, width: int, height: int, bpp: int) -> bytearray:
    """Reverse PNG scanline filtering (spec filter types 0-4) into raw pixel bytes."""
    stride = width * bpp
    out = bytearray(stride * height)
    pos = 0
    for y in range(height):
        filter_type = raw[pos]
        pos += 1
        row = raw[pos : pos + stride]
        pos += stride
        row_start = y * stride
        prev_start = (y - 1) * stride
        for i in range(stride):
            a = out[row_start + i - bpp] if i >= bpp else 0
            b = out[prev_start + i] if y > 0 else 0
            c = out[prev_start + i - bpp] if y > 0 and i >= bpp else 0
            x = row[i]
            if filter_type == 0:
                val = x
            elif filter_type == 1:
                val = (x + a) & 0xFF
            elif filter_type == 2:
                val = (x + b) & 0xFF
            elif filter_type == 3:
                val = (x + (a + b) // 2) & 0xFF
            elif filter_type == 4:
                val = (x + _paeth(a, b, c)) & 0xFF
            else:
                raise ValueError(f"Unsupported PNG filter type {filter_type}")
            out[row_start + i] = val
    return out


def parse_png(data: bytes) -> PngInfo:
    """Parse a PNG's dimensions and, for 8-bit RGB/RGBA, its raw pixel bytes."""
    if not data.startswith(_SIGNATURE):
        raise ValueError("Not a PNG file (bad signature)")

    pos = len(_SIGNATURE)
    width = height = bit_depth = color_type = 0
    idat = bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        tag = data[pos + 4 : pos + 8]
        chunk_data = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk_data[:10])
        elif tag == b"IDAT":
            idat.extend(chunk_data)
        elif tag == b"IEND":
            break

    channels = _SUPPORTED_COLOR_TYPES.get(color_type)
    if channels is None or bit_depth != 8:
        # Dimensions are always real (straight from IHDR); pixel stats aren't
        # implemented here for palette/16-bit/grayscale/interlaced PNGs.
        return PngInfo(width=width, height=height, color_type=color_type, channels=0, pixels=None)

    raw = zlib.decompress(bytes(idat))
    pixels = bytes(_defilter(raw, width, height, channels))
    return PngInfo(
        width=width, height=height, color_type=color_type, channels=channels, pixels=pixels
    )


RGB = tuple[int, int, int]

_NAMED_COLORS: dict[str, RGB] = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
    "red": (220, 20, 20),
    "orange": (255, 140, 0),
    "yellow": (230, 220, 20),
    "green": (34, 139, 34),
    "cyan": (0, 180, 180),
    "blue": (0, 102, 204),
    "purple": (128, 0, 128),
    "pink": (255, 105, 180),
    "brown": (139, 69, 19),
}


def nearest_color_name(rgb: RGB) -> str:
    """The closest named color to `rgb` by squared Euclidean distance.

    A small, fixed reference palette -- not a learned model -- so this is
    a deterministic, explainable approximation, not a claim of true color
    naming. Useful because raw "rgb(30, 90, 200)" tuples don't share any
    words with a query like "which chart is mostly blue".
    """

    def distance(reference: RGB) -> int:
        return sum((a - b) ** 2 for a, b in zip(rgb, reference, strict=True))

    return min(_NAMED_COLORS, key=lambda name: distance(_NAMED_COLORS[name]))


def dominant_colors(info: PngInfo, background_threshold: int = 240) -> tuple[RGB, RGB]:
    """Return (overall dominant color, dominant non-background color) as (r, g, b) tuples.

    Most sample charts are mostly white background, so the overall mode is
    often the background itself -- still an honest, real answer, just not
    the interesting one. The non-background mode (excluding near-white
    pixels) captures the actual "ink" color, which is the more useful
    signal for retrieval and for a human reading the caption.
    """
    if info.pixels is None:
        raise ValueError(
            "No pixel data to compute colors from (unsupported PNG color type/bit depth)"
        )

    counts: Counter[RGB] = Counter()
    non_bg_counts: Counter[RGB] = Counter()
    step = info.channels
    for i in range(0, len(info.pixels), step):
        rgb = (info.pixels[i], info.pixels[i + 1], info.pixels[i + 2])
        counts[rgb] += 1
        if not all(c >= background_threshold for c in rgb):
            non_bg_counts[rgb] += 1

    overall = counts.most_common(1)[0][0]
    non_bg = non_bg_counts.most_common(1)[0][0] if non_bg_counts else overall
    return overall, non_bg


def guess_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    raise ValueError(f"Unsupported image type: {suffix}")


def build_image_content_block(image_bytes: bytes, media_type: str) -> dict:
    """The Anthropic image content block: real base64 pixels, not a caption.

    This is called at generation time, from the actual file on disk, so the
    model sees the real image -- captions only ever drive retrieval.
    """
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.standard_b64encode(image_bytes).decode("ascii"),
        },
    }
