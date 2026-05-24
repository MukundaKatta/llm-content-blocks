"""Tests for llm_content_blocks.Blocks."""

from __future__ import annotations

import base64

import pytest

from llm_content_blocks import (
    VALID_DOCUMENT_MEDIA_TYPES,
    VALID_IMAGE_MEDIA_TYPES,
    Blocks,
)


# ---- text --------------------------------------------------------------


def test_text_basic():
    out = Blocks().text("hello").build()
    assert out == [{"type": "text", "text": "hello"}]


def test_text_cache_control():
    out = Blocks().text("x", cache_control="ephemeral").build()
    assert out[0]["cache_control"] == {"type": "ephemeral"}


def test_text_no_cache_control_by_default():
    out = Blocks().text("x").build()
    assert "cache_control" not in out[0]


# ---- image_b64 ---------------------------------------------------------


def test_image_b64_with_bytes():
    out = Blocks().image_b64(b"\x89PNG", "image/png").build()
    assert out[0]["type"] == "image"
    src = out[0]["source"]
    assert src["type"] == "base64"
    assert src["media_type"] == "image/png"
    assert base64.b64decode(src["data"]) == b"\x89PNG"


def test_image_b64_with_pre_encoded_string():
    encoded = base64.b64encode(b"hi").decode("ascii")
    out = Blocks().image_b64(encoded, "image/png").build()
    assert out[0]["source"]["data"] == encoded


def test_image_b64_rejects_unknown_media_type():
    with pytest.raises(ValueError):
        Blocks().image_b64(b"x", "image/tiff")


def test_image_b64_cache_control():
    out = Blocks().image_b64(b"x", "image/png", cache_control="ephemeral").build()
    assert out[0]["cache_control"] == {"type": "ephemeral"}


def test_valid_image_media_types_set():
    assert "image/png" in VALID_IMAGE_MEDIA_TYPES
    assert "image/jpeg" in VALID_IMAGE_MEDIA_TYPES
    assert "image/webp" in VALID_IMAGE_MEDIA_TYPES
    assert "image/gif" in VALID_IMAGE_MEDIA_TYPES


# ---- image_url ---------------------------------------------------------


def test_image_url_block_shape():
    out = Blocks().image_url("https://example.com/x.png").build()
    assert out == [{"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}}]


# ---- tool_use ----------------------------------------------------------


def test_tool_use_shape():
    out = Blocks().tool_use("toolu_1", "search", {"q": "anthropic"}).build()
    assert out == [{
        "type": "tool_use",
        "id": "toolu_1",
        "name": "search",
        "input": {"q": "anthropic"},
    }]


def test_tool_use_copies_input_dict():
    inp = {"q": "x"}
    out = Blocks().tool_use("u1", "search", inp).build()
    inp["q"] = "MUTATED"
    assert out[0]["input"] == {"q": "x"}


# ---- tool_result -------------------------------------------------------


def test_tool_result_block_via_builder():
    out = Blocks().tool_result_block("u1", "answer").build()
    assert out == [{"type": "tool_result", "tool_use_id": "u1", "content": "answer"}]


def test_tool_result_block_is_error():
    out = Blocks().tool_result_block("u1", "boom", is_error=True).build()
    assert out[0]["is_error"] is True


def test_tool_result_static_one_shot():
    block = Blocks.tool_result("u1", "answer")
    assert block == {"type": "tool_result", "tool_use_id": "u1", "content": "answer"}


def test_tool_result_static_is_error():
    block = Blocks.tool_result("u1", "boom", is_error=True)
    assert block["is_error"] is True


def test_tool_result_omits_is_error_when_false():
    block = Blocks.tool_result("u1", "ok", is_error=False)
    assert "is_error" not in block


# ---- document ---------------------------------------------------------


def test_document_b64_default_pdf():
    out = Blocks().document_b64(b"%PDF-").build()
    assert out[0]["type"] == "document"
    assert out[0]["source"]["media_type"] == "application/pdf"


def test_document_b64_rejects_unknown_media_type():
    with pytest.raises(ValueError):
        Blocks().document_b64(b"x", "application/xml")


def test_valid_document_media_types_set():
    assert "application/pdf" in VALID_DOCUMENT_MEDIA_TYPES


# ---- composition / extend --------------------------------------------


def test_chain_multiple_blocks():
    out = (
        Blocks()
        .text("Look:")
        .image_b64(b"\x89PNG", "image/png")
        .text("done")
        .build()
    )
    assert [b["type"] for b in out] == ["text", "image", "text"]


def test_extend_from_existing_blocks():
    existing = [{"type": "text", "text": "from-elsewhere"}]
    out = Blocks().text("first").extend(existing).build()
    assert len(out) == 2
    assert out[1]["text"] == "from-elsewhere"


def test_extend_copies_each_block():
    existing = [{"type": "text", "text": "x"}]
    b = Blocks().extend(existing)
    existing[0]["text"] = "MUTATED"
    assert b.build()[0]["text"] == "x"


# ---- build / build_message -------------------------------------------


def test_build_returns_copy():
    builder = Blocks().text("x")
    out = builder.build()
    out[0]["text"] = "MUTATED"
    # builder unchanged
    again = builder.build()
    assert again[0]["text"] == "x"


def test_build_message_user_shape():
    msg = Blocks().text("Hi").build_message("user")
    assert msg == {"role": "user", "content": [{"type": "text", "text": "Hi"}]}


def test_build_message_assistant_shape():
    msg = Blocks().text("OK").tool_use("u", "search", {"q": "x"}).build_message("assistant")
    assert msg["role"] == "assistant"
    assert len(msg["content"]) == 2


def test_len_reflects_blocks():
    builder = Blocks()
    assert len(builder) == 0
    builder.text("a").text("b")
    assert len(builder) == 2
