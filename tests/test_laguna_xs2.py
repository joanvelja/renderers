"""Laguna-XS.2-specific renderer behavior.

The Jinja template never iterates ``message.content`` as a list — it
coerces non-string content to ``""``. The renderer, however, is the seam
where a parse → reserialize → re-render round-trip can hand it
structured ``content=[ThinkingPart, TextPart]`` (e.g. trajectory storage
that preserves typed parts). Dropping those parts would silently lose
reasoning from previous turns on every re-render — the regression this
file guards against.
"""

from __future__ import annotations

from functools import lru_cache

import pytest


_MODEL = "poolside/Laguna-XS.2"


@lru_cache(maxsize=1)
def _renderer():
    from renderers import create_renderer
    from renderers.base import load_tokenizer

    return create_renderer(load_tokenizer(_MODEL))


PROMPT = [{"role": "user", "content": "Hi"}]


def test_thinking_part_round_trip_matches_flat_form():
    """``content=[ThinkingPart, TextPart]`` must render identically to
    ``reasoning_content=... , content=...`` — otherwise reasoning is
    silently dropped on every re-render through the trajectory loop."""
    r = _renderer()
    structured = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "thinking through the problem"},
            {"type": "text", "text": "Hello!"},
        ],
    }
    flat = {
        "role": "assistant",
        "reasoning_content": "thinking through the problem",
        "content": "Hello!",
    }
    assert r.render_ids(PROMPT + [structured]) == r.render_ids(PROMPT + [flat])


def test_text_part_only_matches_string_content():
    """List with only ``TextPart`` entries collapses to the equivalent string content."""
    r = _renderer()
    listed = {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]}
    string = {"role": "assistant", "content": "Hello!"}
    assert r.render_ids(PROMPT + [listed]) == r.render_ids(PROMPT + [string])


def test_reasoning_field_beats_thinking_part():
    """Explicit ``reasoning_content`` wins over ``ThinkingPart`` in the
    same message — the field is canonical, the part is the fallback."""
    r = _renderer()
    both = {
        "role": "assistant",
        "reasoning_content": "from field",
        "content": [
            {"type": "thinking", "thinking": "from part (should be ignored)"},
            {"type": "text", "text": "Hi"},
        ],
    }
    field_only = {
        "role": "assistant",
        "reasoning_content": "from field",
        "content": "Hi",
    }
    assert r.render_ids(PROMPT + [both]) == r.render_ids(PROMPT + [field_only])


def test_multiple_thinking_parts_concatenated():
    """Multiple ``ThinkingPart`` entries concatenate (insertion order)."""
    r = _renderer()
    multi = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "first"},
            {"type": "thinking", "thinking": "second"},
            {"type": "text", "text": "out"},
        ],
    }
    flat = {
        "role": "assistant",
        "reasoning_content": "firstsecond",
        "content": "out",
    }
    assert r.render_ids(PROMPT + [multi]) == r.render_ids(PROMPT + [flat])


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(None, id="none"),
        pytest.param([], id="empty-list"),
        pytest.param([{"type": "image"}], id="non-text-non-thinking-only"),
    ],
)
def test_degenerate_content_collapses_to_empty(shape):
    """Content shapes that produce no visible text and no thinking must
    render the same as ``content=""`` — no crashes, no extra tokens."""
    r = _renderer()
    degen = {"role": "assistant", "content": shape}
    empty = {"role": "assistant", "content": ""}
    assert r.render_ids(PROMPT + [degen]) == r.render_ids(PROMPT + [empty])
