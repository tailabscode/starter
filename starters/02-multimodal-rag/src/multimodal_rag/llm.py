"""Model-provider seam: swap between the real Claude client and a
deterministic offline stub without touching any calling code.

Two methods matter here: caption_image (used once per image at ingest
time, cached to disk -- see caption_cache.py) and generate_answer (used
per query). get_client() defaults to AnthropicClient when
ANTHROPIC_API_KEY is set, falls back to StubClient otherwise (logging
why), and --offline always forces the stub. Tests use StubClient
exclusively and never touch the network.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Protocol

import anthropic

from .caption_cache import ImageCaption
from .config import Config
from .generation import SYSTEM_PROMPT, ContentItem, build_content_blocks
from .images import dominant_colors, nearest_color_name, parse_png

logger = logging.getLogger(__name__)

CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {
            "type": "string",
            "description": "A short, searchable description of what the image shows.",
        },
        "extracted_text": {
            "type": "string",
            "description": "Any text visible in the image, transcribed verbatim. Empty if none.",
        },
    },
    "required": ["caption", "extracted_text"],
    "additionalProperties": False,
}


class LLMClient(Protocol):
    provider: str

    def caption_image(self, image_bytes: bytes, media_type: str) -> ImageCaption: ...
    def generate_answer(self, query: str, content_items: list[ContentItem]) -> str: ...


class AnthropicClient:
    """Real Claude client: vision captioning at ingest, grounded answers at query."""

    provider = "anthropic"

    def __init__(self, model: str) -> None:
        self._client = anthropic.Anthropic()
        self._model = model

    def caption_image(self, image_bytes: bytes, media_type: str) -> ImageCaption:
        image_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.standard_b64encode(image_bytes).decode("ascii"),
            },
        }
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            image_block,
                            {
                                "type": "text",
                                "text": (
                                    "Describe this image for a search index, and "
                                    "transcribe any visible text."
                                ),
                            },
                        ],
                    }
                ],
                output_config={"format": {"type": "json_schema", "schema": CAPTION_SCHEMA}},
            )
        except anthropic.AuthenticationError as exc:
            raise RuntimeError(
                "Anthropic API rejected the API key. Check ANTHROPIC_API_KEY in .env."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise RuntimeError("Anthropic API rate limit hit. Wait and retry.") from exc
        except anthropic.APIStatusError as exc:
            raise RuntimeError(f"Anthropic API returned an error: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise RuntimeError(f"Could not reach the Anthropic API: {exc}") from exc

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        data = json.loads(text)
        return ImageCaption(caption=data["caption"], extracted_text=data["extracted_text"])

    def generate_answer(self, query: str, content_items: list[ContentItem]) -> str:
        blocks = build_content_blocks(content_items)
        blocks.append({"type": "text", "text": f"Question: {query}"})
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": blocks}],
            )
        except anthropic.AuthenticationError as exc:
            raise RuntimeError(
                "Anthropic API rejected the API key. Check ANTHROPIC_API_KEY in .env."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise RuntimeError("Anthropic API rate limit hit. Wait and retry.") from exc
        except anthropic.APIStatusError as exc:
            raise RuntimeError(f"Anthropic API returned an error: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise RuntimeError(f"Could not reach the Anthropic API: {exc}") from exc
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )


class StubClient:
    """Deterministic, offline stand-in for AnthropicClient.

    caption_image genuinely decodes the PNG's pixels (dimensions + two
    real color statistics, see images.py) -- it is not a lorem-ipsum
    placeholder, but it also cannot understand image *content* the way
    vision can, and it never claims to. generate_answer echoes the actual
    retrieved items (their real caption/chunk text), it does not look at
    image pixels a second time. Used for --offline runs and tests.
    """

    provider = "stub"
    SNIPPET_CHARS = 220

    def caption_image(self, image_bytes: bytes, media_type: str) -> ImageCaption:
        if media_type != "image/png":
            return ImageCaption(
                caption=f"Image ({media_type}); offline pixel inspection only supports PNG.",
                extracted_text="",
            )
        info = parse_png(image_bytes)
        if info.pixels is None:
            caption = (
                f"{info.width}x{info.height}px image (PNG color type {info.color_type}); "
                "offline pixel-color inspection only supports 8-bit RGB/RGBA PNGs."
            )
            return ImageCaption(caption=caption, extracted_text="")

        overall, non_bg = dominant_colors(info)
        overall_name, non_bg_name = nearest_color_name(overall), nearest_color_name(non_bg)
        caption = (
            f"{info.width}x{info.height}px image. "
            f"Dominant color overall: {overall_name} (rgb{overall}), often the background. "
            f"Dominant non-background color: {non_bg_name} (rgb{non_bg})."
        )
        return ImageCaption(caption=caption, extracted_text="")

    def generate_answer(self, query: str, content_items: list[ContentItem]) -> str:
        if not content_items:
            return "[stub] I don't know -- no context was retrieved for this question."
        lines = [f"[stub] Grounded answer for: {query!r}"]
        for ci in content_items:
            item = ci.item
            label = "image" if item.kind == "image" else "text"
            snippet = " ".join(item.text.split())[: self.SNIPPET_CHARS]
            lines.append(f"({label} {item.source}) {snippet}... [{ci.number}]")
        return " ".join(lines)


def get_client(config: Config, offline: bool) -> LLMClient:
    """Pick StubClient or AnthropicClient per --offline / LLM_PROVIDER / API key."""
    if offline:
        return StubClient()

    provider = config.llm_provider
    if provider == "stub":
        return StubClient()
    if provider == "anthropic":
        if not config.anthropic_api_key:
            logger.info(
                "ANTHROPIC_API_KEY not set; falling back to StubClient. "
                "Set ANTHROPIC_API_KEY in .env, or LLM_PROVIDER=stub to silence this."
            )
            return StubClient()
        return AnthropicClient(model=config.anthropic_model)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (expected 'anthropic' or 'stub')")
