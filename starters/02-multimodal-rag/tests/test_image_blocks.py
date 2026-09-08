"""PNG pixel decoding and Anthropic image content-block construction."""

from __future__ import annotations

import base64
from pathlib import Path

from conftest import make_png_bytes

from multimodal_rag.generation import build_content_blocks, build_content_items
from multimodal_rag.images import (
    PngInfo,
    build_image_content_block,
    dominant_colors,
    guess_media_type,
    nearest_color_name,
    parse_png,
)
from multimodal_rag.items import Item


def test_parse_png_reads_real_dimensions_and_pixels() -> None:
    data = make_png_bytes(4, 3, (10, 20, 30))
    info = parse_png(data)
    assert (info.width, info.height) == (4, 3)
    assert info.pixels is not None
    assert len(info.pixels) == 4 * 3 * 3
    # Every pixel should be exactly the solid color we encoded.
    assert info.pixels[:3] == bytes((10, 20, 30))
    assert info.pixels[-3:] == bytes((10, 20, 30))


def test_dominant_colors_finds_the_real_solid_color() -> None:
    data = make_png_bytes(5, 5, (200, 30, 30))
    info = parse_png(data)
    overall, non_bg = dominant_colors(info)
    assert overall == (200, 30, 30)
    assert non_bg == (200, 30, 30)


def test_dominant_colors_excludes_near_white_background() -> None:
    # A single non-white pixel in a mostly-white image: the overall mode is
    # white, but the non-background mode must be the real accent color.
    data = make_png_bytes(3, 3, (255, 255, 255))
    info = parse_png(data)
    # Manually flip one pixel to blue by re-encoding (keeps the test honest
    # about what dominant_colors actually computes from real bytes).
    pixels = bytearray(info.pixels)
    pixels[0:3] = bytes((0, 0, 200))
    patched = PngInfo(
        width=info.width,
        height=info.height,
        color_type=info.color_type,
        channels=3,
        pixels=bytes(pixels),
    )
    overall, non_bg = dominant_colors(patched)
    assert overall == (255, 255, 255)
    assert non_bg == (0, 0, 200)


def test_nearest_color_name_is_deterministic_and_reasonable() -> None:
    assert nearest_color_name((0, 102, 204)) == "blue"
    assert nearest_color_name((34, 139, 34)) == "green"
    assert nearest_color_name((30, 90, 200)) == "blue"  # our sample chart's real blue
    assert nearest_color_name((255, 255, 255)) == "white"
    assert nearest_color_name((0, 0, 0)) == "black"


def test_guess_media_type_for_png() -> None:
    assert guess_media_type(Path("chart.png")) == "image/png"


def test_build_image_content_block_has_correct_media_type_and_base64() -> None:
    image_bytes = make_png_bytes(2, 2, (1, 2, 3))
    block = build_image_content_block(image_bytes, "image/png")

    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"
    # The base64 payload must decode back to the exact original bytes.
    decoded = base64.standard_b64decode(block["source"]["data"])
    assert decoded == image_bytes


def test_build_content_blocks_interleaves_text_label_and_image_block(tmp_path: Path) -> None:
    image_path = tmp_path / "pic.png"
    image_bytes = make_png_bytes(2, 2, (9, 9, 9))
    image_path.write_bytes(image_bytes)

    items = [
        Item(item_id="t1", kind="text", source="doc.md", text="hello", heading_trail=["H"]),
        Item(
            item_id="i1",
            kind="image",
            source="pic.png",
            text="a caption",
            heading_trail=[],
            image_path=str(image_path),
            media_type="image/png",
        ),
    ]
    content_items = build_content_items(items)
    blocks = build_content_blocks(content_items)

    # text item -> one text block; image item -> a label text block + an image block.
    assert len(blocks) == 3
    assert blocks[0]["type"] == "text" and "[1]" in blocks[0]["text"]
    assert blocks[1]["type"] == "text"
    assert "[2]" in blocks[1]["text"] and "pic.png" in blocks[1]["text"]
    assert blocks[2]["type"] == "image"
    decoded = base64.standard_b64decode(blocks[2]["source"]["data"])
    assert decoded == image_bytes
