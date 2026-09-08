"""Generate the tiny sample PNGs used by this starter's demo corpus.

A from-scratch PNG encoder using only zlib + struct -- no Pillow, no
numpy. Each image is deliberately simple (flat-colored bars or boxes on a
white background) so the offline StubClient's pixel-derived captions
(dimensions + dominant color, see src/multimodal_rag/images.py) are
genuinely informative rather than just plausible-sounding.

Re-run this script to regenerate the images in data/corpus/images/:

    python data/make_samples.py
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

Color = tuple[int, int, int]

OUT_DIR = Path(__file__).parent / "corpus" / "images"
WIDTH, HEIGHT = 100, 70


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def encode_png(width: int, height: int, pixels: bytearray) -> bytes:
    """Encode raw 8-bit RGB pixel data (row-major, top-to-bottom) as a PNG."""
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit, color type 2 (RGB)
    stride = width * 3
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0 (None) for every scanline
        raw.extend(pixels[y * stride : (y + 1) * stride])
    idat = zlib.compress(bytes(raw), 9)
    return signature + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def blank_canvas(width: int, height: int, background: Color = (255, 255, 255)) -> bytearray:
    return bytearray(bytes(background) * (width * height))


def fill_rect(
    pixels: bytearray, width: int, x0: int, y0: int, x1: int, y1: int, color: Color
) -> None:
    for y in range(y0, y1):
        start = (y * width + x0) * 3
        end = (y * width + x1) * 3
        pixels[start:end] = bytes(color) * (x1 - x0)


def bar_chart(main_color: Color, accent_color: Color, bar_heights: list[int]) -> bytearray:
    """A simple vertical bar chart: three bars in main_color, one accent bar."""
    pixels = blank_canvas(WIDTH, HEIGHT)
    baseline = HEIGHT - 6
    bar_width, gap = 14, 8
    x = 8
    colors = [main_color, main_color, accent_color, main_color]
    for bar_height, color in zip(bar_heights, colors, strict=True):
        fill_rect(pixels, WIDTH, x, baseline - bar_height, x + bar_width, baseline, color)
        x += bar_width + gap
    return pixels


def schematic() -> bytearray:
    """Three connected boxes: a minimal ingest-pipeline schematic."""
    pixels = blank_canvas(WIDTH, HEIGHT)
    box_color = (90, 90, 90)
    line_color = (20, 20, 20)
    box_w, box_h = 20, 24
    y0 = (HEIGHT - box_h) // 2
    right_edges: list[tuple[int, int]] = []
    for x0 in (6, 40, 74):
        fill_rect(pixels, WIDTH, x0, y0, x0 + box_w, y0 + box_h, box_color)
        right_edges.append((x0 + box_w, y0 + box_h // 2))
    for (x_start, y_mid), x0_next in zip(right_edges[:-1], (40, 74), strict=True):
        fill_rect(pixels, WIDTH, x_start, y_mid - 1, x0_next, y_mid + 1, line_color)
    return pixels


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images = {
        "chart_q1_blue.png": bar_chart((30, 90, 200), (230, 140, 20), [30, 45, 20, 55]),
        "chart_q2_green.png": bar_chart((40, 160, 70), (200, 40, 40), [50, 25, 40, 35]),
        "chart_q3_orange.png": bar_chart((230, 140, 20), (30, 90, 200), [20, 55, 45, 30]),
        "schematic_pipeline.png": schematic(),
    }

    for name, pixels in images.items():
        png_bytes = encode_png(WIDTH, HEIGHT, pixels)
        out_path = OUT_DIR / name
        out_path.write_bytes(png_bytes)
        print(f"{name}: {len(png_bytes)} bytes -> {out_path}")


if __name__ == "__main__":
    main()
