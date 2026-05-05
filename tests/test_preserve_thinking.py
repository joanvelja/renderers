"""Smoke coverage for the ``preserve_*_thinking`` override flags.

Two invariants per renderer:

1. Default render (both flags ``False``) is byte-identical to the existing
   ``apply_chat_template`` parity baseline — covered exhaustively elsewhere.
2. Setting either flag never *removes* tokens compared to the default and,
   for renderers whose template would drop past-asst thinking, actually
   adds tokens for a conversation containing past-asst ``reasoning_content``.

Renderers whose template either always preserves thinking (DeepSeek-V3) or
never references ``reasoning_content`` for past-asst (Kimi-K2, Qwen3-VL)
are no-ops by design — they're listed below and the test asserts the
default==override equality instead of strict growth.
"""

from __future__ import annotations

import pytest

from renderers.base import should_preserve_past_thinking


# Renderers whose template doesn't drop past-asst thinking or has no
# place to re-emit it. For these, override flags MUST be no-ops.
NO_OP_MODELS = {
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V3-Base",
    "moonshotai/Kimi-K2-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
}


CONVERSATION = [
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "reasoning_content": "I should call the weather tool for Paris.",
        "content": "Let me check.",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        ],
    },
    {"role": "tool", "name": "get_weather", "content": "Sunny, 22C"},
    {
        "role": "assistant",
        "reasoning_content": "The tool returned the weather.",
        "content": "Sunny, 22C in Paris.",
    },
    {"role": "user", "content": "And Berlin?"},
]


def test_should_preserve_past_thinking_classification():
    # Asst[1] has tool_calls and is followed by tool[2] — between-tool-calls.
    assert should_preserve_past_thinking(
        CONVERSATION,
        1,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=True,
    )
    # Asst[3] is followed by user[4] — not between tool calls.
    assert not should_preserve_past_thinking(
        CONVERSATION,
        3,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=True,
    )
    # preserve_all_thinking applies to every past-asst regardless.
    assert should_preserve_past_thinking(
        CONVERSATION,
        3,
        preserve_all_thinking=True,
        preserve_thinking_between_tool_calls=False,
    )
    # Both flags False → always False.
    assert not should_preserve_past_thinking(
        CONVERSATION,
        1,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=False,
    )


def test_preserve_flags_default_unchanged(model_name, tokenizer, renderer):
    # Calling render with the flags explicitly off must be byte-identical
    # to the bare call (defaults).
    bare = renderer.render_ids(CONVERSATION)
    explicit_off = renderer.render_ids(
        CONVERSATION,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=False,
    )
    assert bare == explicit_off, (
        f"{model_name}: explicit flags=False must equal bare default render"
    )


def test_preserve_all_thinking_grows_or_no_op(model_name, tokenizer, renderer):
    from renderers.default import DefaultRenderer

    if isinstance(renderer, DefaultRenderer):
        pytest.skip("DefaultRenderer raises on these flags — covered separately")
    default = renderer.render_ids(CONVERSATION)
    preserved = renderer.render_ids(CONVERSATION, preserve_all_thinking=True)

    if model_name in NO_OP_MODELS:
        assert preserved == default, (
            f"{model_name} is a no-op renderer; preserve_all_thinking must "
            f"not change output (got {len(default)} → {len(preserved)})"
        )
    else:
        assert len(preserved) > len(default), (
            f"{model_name}: preserve_all_thinking should add tokens for a "
            f"conversation with past-asst reasoning_content "
            f"(default={len(default)}, preserved={len(preserved)})"
        )


def test_preserve_between_tool_calls_strict_subset(model_name, tokenizer, renderer):
    """``preserve_thinking_between_tool_calls`` is strictly weaker than
    ``preserve_all_thinking``: token count satisfies default <= between <= all."""
    from renderers.default import DefaultRenderer

    if isinstance(renderer, DefaultRenderer):
        pytest.skip("DefaultRenderer raises on these flags — covered separately")
    default = renderer.render_ids(CONVERSATION)
    between = renderer.render_ids(
        CONVERSATION, preserve_thinking_between_tool_calls=True
    )
    all_ = renderer.render_ids(CONVERSATION, preserve_all_thinking=True)
    assert len(default) <= len(between) <= len(all_), (
        f"{model_name}: expected default <= between <= all, "
        f"got {len(default)} <= {len(between)} <= {len(all_)}"
    )


def test_default_renderer_raises_on_flags():
    """``DefaultRenderer`` falls back to apply_chat_template with no
    selective re-emit pathway, so it must raise rather than silently ignore."""
    from transformers import AutoTokenizer

    from renderers import create_renderer

    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
    )
    renderer = create_renderer(tok, renderer="default")
    with pytest.raises(NotImplementedError):
        renderer.render_ids(CONVERSATION, preserve_all_thinking=True)
    with pytest.raises(NotImplementedError):
        renderer.render_ids(CONVERSATION, preserve_thinking_between_tool_calls=True)
