"""llm-content-blocks - typed builder for Anthropic content blocks.

The Anthropic Messages API takes a list of content blocks per message:
text, image, tool_use, tool_result, document. The shapes are simple but
fiddly, especially when you stitch them together programmatically.
`Blocks()` is a fluent builder that emits the exact dict shape the API
expects, with no SDK dependency.

    from llm_content_blocks import Blocks

    content = (
        Blocks()
        .text("Look at this:")
        .image_b64(data, "image/png")
        .text("What is it?", cache_control="ephemeral")
        .build()
    )

    user_msg = Blocks().text("Hi").build_message("user")
    # -> {"role": "user", "content": [{"type": "text", "text": "Hi"}]}

    tool_result = Blocks.tool_result("toolu_1", "the answer", is_error=False)
    # -> {"type": "tool_result", "tool_use_id": "toolu_1", "content": "the answer"}

Useful when you assemble messages on the fly: tool responses, multi-image
inputs, cached system prompts, retry-with-correction loops. The class is
intentionally light — it does not validate against the live API, it just
makes the right shape.
"""

from __future__ import annotations

import base64
from typing import Any, Iterable, Literal

__version__ = "0.1.0"
__all__ = [
    "Blocks",
    "VALID_IMAGE_MEDIA_TYPES",
    "VALID_DOCUMENT_MEDIA_TYPES",
]


VALID_IMAGE_MEDIA_TYPES = frozenset({
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
})

VALID_DOCUMENT_MEDIA_TYPES = frozenset({
    "application/pdf",
    "text/plain",
})


CacheControl = Literal["ephemeral"] | None


def _cc(cache_control: CacheControl) -> dict | None:
    if cache_control is None:
        return None
    return {"type": cache_control}


class Blocks:
    """Fluent builder for a list of Anthropic content blocks."""

    def __init__(self) -> None:
        self._blocks: list[dict[str, Any]] = []

    # ---- chained appenders -----------------------------------------

    def text(
        self,
        s: str,
        *,
        cache_control: CacheControl = None,
    ) -> "Blocks":
        block: dict[str, Any] = {"type": "text", "text": s}
        cc = _cc(cache_control)
        if cc is not None:
            block["cache_control"] = cc
        self._blocks.append(block)
        return self

    def image_b64(
        self,
        data: bytes | str,
        media_type: str,
        *,
        cache_control: CacheControl = None,
    ) -> "Blocks":
        if media_type not in VALID_IMAGE_MEDIA_TYPES:
            raise ValueError(
                f"unsupported image media_type {media_type!r}; "
                f"expected one of {sorted(VALID_IMAGE_MEDIA_TYPES)}"
            )
        if isinstance(data, bytes):
            encoded = base64.b64encode(data).decode("ascii")
        else:
            encoded = data
        block: dict[str, Any] = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": encoded,
            },
        }
        cc = _cc(cache_control)
        if cc is not None:
            block["cache_control"] = cc
        self._blocks.append(block)
        return self

    def image_url(
        self,
        url: str,
        *,
        cache_control: CacheControl = None,
    ) -> "Blocks":
        block: dict[str, Any] = {
            "type": "image",
            "source": {"type": "url", "url": url},
        }
        cc = _cc(cache_control)
        if cc is not None:
            block["cache_control"] = cc
        self._blocks.append(block)
        return self

    def tool_use(
        self,
        id: str,
        name: str,
        input: dict[str, Any],
    ) -> "Blocks":
        self._blocks.append({
            "type": "tool_use",
            "id": id,
            "name": name,
            "input": dict(input),
        })
        return self

    def tool_result_block(
        self,
        tool_use_id: str,
        content: Any,
        *,
        is_error: bool = False,
    ) -> "Blocks":
        """Append a tool_result block (chained form of `Blocks.tool_result`)."""
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            block["is_error"] = True
        self._blocks.append(block)
        return self

    def document_b64(
        self,
        data: bytes | str,
        media_type: str = "application/pdf",
        *,
        cache_control: CacheControl = None,
    ) -> "Blocks":
        if media_type not in VALID_DOCUMENT_MEDIA_TYPES:
            raise ValueError(
                f"unsupported document media_type {media_type!r}; "
                f"expected one of {sorted(VALID_DOCUMENT_MEDIA_TYPES)}"
            )
        if isinstance(data, bytes):
            encoded = base64.b64encode(data).decode("ascii")
        else:
            encoded = data
        block: dict[str, Any] = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": encoded,
            },
        }
        cc = _cc(cache_control)
        if cc is not None:
            block["cache_control"] = cc
        self._blocks.append(block)
        return self

    def extend(self, blocks: Iterable[dict[str, Any]]) -> "Blocks":
        """Splice an existing iterable of content blocks into the builder."""
        self._blocks.extend(dict(b) for b in blocks)
        return self

    # ---- terminal ---------------------------------------------------

    def build(self) -> list[dict[str, Any]]:
        """Return a *copy* of the accumulated content-block list."""
        return [dict(b) for b in self._blocks]

    def build_message(self, role: Literal["user", "assistant"]) -> dict[str, Any]:
        """Wrap the blocks into `{"role": role, "content": [...]}`."""
        return {"role": role, "content": self.build()}

    def __len__(self) -> int:
        return len(self._blocks)

    # ---- static one-shots ------------------------------------------

    @staticmethod
    def tool_result(
        tool_use_id: str,
        content: Any,
        *,
        is_error: bool = False,
    ) -> dict[str, Any]:
        """Return a single tool_result block (not wrapped in a list)."""
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            block["is_error"] = True
        return block
