"""Kimi K2.5 tool-schema rendering follows the HF compact-JSON branch."""

from __future__ import annotations

import json

from renderers import create_renderer
from renderers.base import load_tokenizer


def test_tool_schema_renders_as_compact_json_with_json_literals():
    """Tool declarations must mirror ``tools | tojson(separators=(',', ':'))``."""
    tokenizer = load_tokenizer("moonshotai/Kimi-K2.5")
    renderer = create_renderer(tokenizer)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "set_mode",
                "description": "Configure mode.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean", "default": True},
                        "mode": {"enum": ["auto", None, True, False, 0]},
                    },
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "configure mode"}]

    text = tokenizer.decode(renderer.render_ids(messages, tools=tools))
    expected = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))

    assert f"<|im_middle|>{expected}<|im_end|>" in text
    assert '"default":true' in text
    assert '"mode":{"enum":["auto",null,true,false,0]}' in text
    assert "Default: True" not in text
    assert '"default": True' not in text
