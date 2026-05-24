# llm-content-blocks

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/llm-content-blocks.svg)](https://pypi.org/project/llm-content-blocks/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Typed builder for Anthropic Messages-API content blocks.** Zero deps, no SDK requirement.

```python
from llm_content_blocks import Blocks

# build a list of content blocks
content = (
    Blocks()
    .text("Look at this image:")
    .image_b64(png_bytes, "image/png")
    .text("What's in it?", cache_control="ephemeral")
    .build()
)

# wrap as a full message
user_msg = Blocks().text("Hi").build_message("user")
# {"role": "user", "content": [{"type": "text", "text": "Hi"}]}

# tool result (one-shot, not chained)
tr = Blocks.tool_result("toolu_1", "the answer")
# {"type": "tool_result", "tool_use_id": "toolu_1", "content": "the answer"}
```

## Why

The Anthropic Messages content-block shapes are simple but fiddly when you assemble them programmatically: tool responses, multi-image inputs, cached system prompts, retry-with-correction loops. `Blocks()` is a fluent builder that emits the exact dict shape the API expects.

No SDK dependency. The library doesn't talk to the API. It just makes the right shape — which means you can use it from any HTTP client (httpx, requests, raw urllib) or pass the output to the official SDK.

Supported blocks: `text`, `image` (base64 or URL), `tool_use`, `tool_result`, `document` (PDF, plain text).

## Install

```bash
pip install llm-content-blocks
```

## API

```python
from llm_content_blocks import Blocks

# chained appends — return self
Blocks().text(s, *, cache_control=None)
Blocks().image_b64(data_bytes_or_b64_str, media_type, *, cache_control=None)
Blocks().image_url(url, *, cache_control=None)
Blocks().tool_use(id, name, input: dict)
Blocks().tool_result_block(tool_use_id, content, *, is_error=False)
Blocks().document_b64(data, media_type="application/pdf", *, cache_control=None)
Blocks().extend(existing_blocks)

# terminal
Blocks().build() -> list[dict]                       # copy of the block list
Blocks().build_message("user" | "assistant") -> dict # {"role", "content"}
len(Blocks())                                        # current block count

# one-shot helpers
Blocks.tool_result(tool_use_id, content, *, is_error=False) -> dict
```

`cache_control="ephemeral"` is the only valid value today (Anthropic's prompt cache). Passing `None` (default) omits the field. The lib rejects unknown image and document media types up front so you get a `ValueError` from the build site, not a 400 from the API.

## Companion libraries

- [`prompt-cache-warmer`](https://github.com/MukundaKatta/prompt-cache-warmer) — warm a system prompt block (built with this lib) before user traffic.
- [`agentprompt-rs`](https://github.com/MukundaKatta/agentprompt-rs) — Jinja2-style prompt template render; pair with this builder for the final message construction.
- [`anthropic-batch-kit`](https://github.com/MukundaKatta/anthropic-batch-kit) — feed messages built here into the Batches API.

## License

MIT
