"""Tests for the pluggable ToolParser / ReasoningParser registry."""

from __future__ import annotations

import json

import pytest

from renderers.base import ToolCallParseStatus, load_tokenizer
from renderers.parsers import (
    REASONING_PARSERS,
    TOOL_PARSERS,
    Qwen3ToolParser,
    ThinkTextReasoningParser,
    get_reasoning_parser,
    get_tool_parser,
)


def test_registries_nonempty():
    assert "qwen3" in TOOL_PARSERS
    assert "qwen3.5" in TOOL_PARSERS
    assert "glm" in TOOL_PARSERS
    assert "deepseek-v3" in TOOL_PARSERS
    assert "think" in REASONING_PARSERS


def test_unknown_parser_errors():
    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    with pytest.raises(ValueError, match="Unknown tool_parser"):
        get_tool_parser("does-not-exist", tok)
    with pytest.raises(ValueError, match="Unknown reasoning_parser"):
        get_reasoning_parser("does-not-exist", tok)


def test_qwen3_tool_parser_roundtrip():
    """Tokenize a Hermes-style tool call, parse it back out."""
    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    parser = get_tool_parser("qwen3", tok)
    assert isinstance(parser, Qwen3ToolParser)

    completion_text = 'hello\n<tool_call>\n{"name": "search", "arguments": {"q": "rain"}}\n</tool_call>'
    token_ids = tok.encode(completion_text, add_special_tokens=False)
    content_ids, tool_calls = parser.extract(list(token_ids))
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc.status == ToolCallParseStatus.OK
    assert tc.name == "search"
    assert tc.arguments == {"q": "rain"}
    assert tc.token_span is not None and tc.token_span[0] < tc.token_span[1]
    # content_ids should cover everything up to (but not including) <tool_call>
    content_text = tok.decode(content_ids, skip_special_tokens=False)
    assert "hello" in content_text
    assert "<tool_call>" not in content_text


def test_qwen3_tool_parser_no_tool_call():
    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    parser = get_tool_parser("qwen3", tok)
    ids = tok.encode("just plain text response", add_special_tokens=False)
    content_ids, tool_calls = parser.extract(list(ids))
    assert tool_calls == []
    assert content_ids == list(ids)


def test_qwen3_tool_parser_records_invalid_json():
    """Malformed JSON in a <tool_call> block surfaces as INVALID_JSON, not silently dropped."""
    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    parser = get_tool_parser("qwen3", tok)
    completion = (
        'hi\n<tool_call>\n{"name": "f", "arguments": {broken json\n</tool_call>'
    )
    ids = tok.encode(completion, add_special_tokens=False)
    _, tool_calls = parser.extract(list(ids))
    assert len(tool_calls) == 1
    assert tool_calls[0].status == ToolCallParseStatus.INVALID_JSON
    assert tool_calls[0].raw  # raw block text preserved


def test_qwen3_tool_parser_parallel_partial_success():
    """Parallel calls: parser keeps the good ones AND records the broken one."""
    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    parser = get_tool_parser("qwen3", tok)
    completion = (
        "pre\n"
        '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
        "<tool_call>\n{broken\n</tool_call>\n"
        '<tool_call>\n{"name": "c", "arguments": {"x": 1}}\n</tool_call>'
    )
    ids = tok.encode(completion, add_special_tokens=False)
    _, tool_calls = parser.extract(list(ids))
    assert [tc.status for tc in tool_calls] == [
        ToolCallParseStatus.OK,
        ToolCallParseStatus.INVALID_JSON,
        ToolCallParseStatus.OK,
    ]
    assert [tc.name for tc in tool_calls] == ["a", None, "c"]


def test_think_reasoning_parser_extracts_block():
    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    parser = get_reasoning_parser("think", tok)
    assert isinstance(parser, ThinkTextReasoningParser)
    reasoning, content = parser.extract("<think>let me think</think>the answer")
    assert reasoning == "let me think"
    assert content == "the answer"


def test_think_reasoning_parser_no_block():
    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    parser = get_reasoning_parser("think", tok)
    reasoning, content = parser.extract("no reasoning here")
    assert reasoning is None
    assert content == "no reasoning here"


def test_default_renderer_uses_parsers():
    """DefaultRenderer + parsers should extract tool calls and reasoning."""
    from renderers import DefaultRendererConfig, create_renderer

    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    renderer = create_renderer(
        tok,
        DefaultRendererConfig(tool_parser="qwen3", reasoning_parser="think"),
    )
    assert renderer.supports_tools is True

    completion = '<think>think</think>ok\n<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>'
    ids = tok.encode(completion, add_special_tokens=False)
    parsed = renderer.parse_response(list(ids))
    assert parsed.reasoning_content == "think"
    assert parsed.content.startswith("ok")
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "f"
    assert parsed.tool_calls[0].status == ToolCallParseStatus.OK


def test_default_renderer_with_a_tool_parser_refuses_a_parse_without_tool_calls():
    """Its vLLM parser pair isn't pinned, so it can't serve a reply as vLLM's chat
    path would under tool_choice="none"; without a tool parser there's no scan to
    skip, and the parse is the same."""
    from renderers import DefaultRendererConfig, create_renderer
    from renderers.base import require_vllm_parsers_for

    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    completion = 'ok\n<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>'
    ids = tok.encode(completion, add_special_tokens=False)
    parsing = create_renderer(tok, DefaultRendererConfig(tool_parser="qwen3"))
    with pytest.raises(ValueError, match="DefaultRenderer parses tool calls but pins"):
        parsing.parse_response(list(ids), tool_calls=False)
    with pytest.raises(ValueError, match="renderer 'default' parses tool calls but"):
        require_vllm_parsers_for(DefaultRendererConfig(tool_parser="qwen3"), "m/x")
    plain = create_renderer(tok, DefaultRendererConfig())
    assert plain.parse_response(list(ids), tool_calls=False) == plain.parse_response(
        list(ids)
    )
    require_vllm_parsers_for(DefaultRendererConfig(), "m/x")
    require_vllm_parsers_for(None, "m/unmapped")


_TEXT_ONLY_RENDERER = """
from typing import Literal

from renderers.configs import BaseRendererConfig


class TextOnlyRendererConfig(BaseRendererConfig):
    name: Literal["text-only"] = "text-only"


class DeclaredTextOnly:
    config_class = TextOnlyRendererConfig
    supports_tools = False

    def __init__(self, tokenizer, config):
        self.config = config


class DerivedTextOnly(DeclaredTextOnly):
    @property
    def supports_tools(self):
        return False
"""


@pytest.mark.asyncio
async def test_tools_on_a_renderer_that_parses_no_tool_calls_are_refused_before_any_request(
    tmp_path,
):
    """A renderer takes tools only if the built renderer parses tool calls
    (``supports_tools``, a class attribute or a property): not DefaultRenderer
    without a tool_parser, which auto falls back to for a model the renderer
    map lacks, nor a custom renderer declaring none either way. generate
    refuses tools for one before any request, with one error naming it and the
    fix."""
    from replay import replay_client

    from renderers import (
        CustomRendererConfig,
        DefaultRendererConfig,
        ToolsUnsupportedError,
        create_renderer,
    )
    from renderers.base import require_tool_support
    from renderers.client import generate
    from renderers.configs import config_from_name

    (tmp_path / "text_only.py").write_text(_TEXT_ONLY_RENDERER)
    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    refused = [DefaultRendererConfig()] + [
        CustomRendererConfig(import_path=f"{tmp_path / 'text_only.py'}:{name}")
        for name in ["DeclaredTextOnly", "DerivedTextOnly"]
    ]
    for config in refused:
        renderer = create_renderer(tok, config)
        requests = []
        async with replay_client([], requests) as client:
            with pytest.raises(
                ToolsUnsupportedError,
                match=f"^{type(renderer).__name__} parses no tool calls, so it "
                r"can't take tools: configure a renderer that parses them \(a "
                "model-specific one, or the default one with a tool_parser\\)$",
            ):
                await generate(
                    client=client,
                    renderer=renderer,
                    model="Qwen/Qwen3-0.6B",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=tools,
                )
        assert requests == []
    for config in [
        config_from_name("qwen3"),
        DefaultRendererConfig(tool_parser="qwen3"),
    ]:
        require_tool_support(create_renderer(tok, config))


def test_a_checkpoint_is_served_its_own_pair_under_the_renderer_it_maps_to(
    monkeypatch,
):
    """Kimi-K2-Instruct answers without thinking, so it is served without a
    reasoning parser, through auto or its own renderer; any other model or
    renderer gets the renderer's pair. An entry may change only the reasoning
    parser: the renderer's parse follows the tool parser's markup rule, so an
    entry changing the tool parser, or the markup rule with a legacy reasoning
    parser, is refused."""
    from renderers import VllmParsers
    from renderers.base import MODEL_VLLM_PARSERS, vllm_parsers_for
    from renderers.configs import config_from_name

    model = "moonshotai/Kimi-K2-Instruct"
    instruct = VllmParsers("kimi_k2", None)
    assert vllm_parsers_for(None, model) == instruct
    assert vllm_parsers_for(config_from_name("kimi-k2"), model) == instruct
    thinking = VllmParsers("kimi_k2", "kimi_k2")
    assert vllm_parsers_for(config_from_name("kimi-k2"), "m/kimi-ft") == thinking
    assert vllm_parsers_for(config_from_name("kimi-k2.5"), model) == thinking
    for entry in [VllmParsers("hermes", None), VllmParsers("kimi_k2", "deepseek_r1")]:
        monkeypatch.setitem(MODEL_VLLM_PARSERS, model, entry)
        with pytest.raises(ValueError, match="only the reasoning parser"):
            vllm_parsers_for(None, model)


def test_an_entry_for_a_checkpoint_the_renderer_map_lacks_is_refused_by_name(
    monkeypatch,
):
    """An entry's pair holds only under the renderer its checkpoint maps to, so
    an entry for a checkpoint ``MODEL_RENDERER_MAP`` lacks is a broken table,
    refused naming the entry and the fix, through auto or a named renderer."""
    from renderers import VllmParsers
    from renderers.base import MODEL_VLLM_PARSERS, vllm_parsers_for
    from renderers.configs import config_from_name

    model = "my-org/unmapped-ckpt"
    monkeypatch.setitem(MODEL_VLLM_PARSERS, model, VllmParsers("hermes", None))
    for config in [None, config_from_name("qwen3")]:
        with pytest.raises(
            ValueError,
            match=r"MODEL_VLLM_PARSERS\['my-org/unmapped-ckpt'\] .* "
            r"MODEL_RENDERER_MAP doesn't map: map it to its renderer, or drop "
            "the entry",
        ):
            vllm_parsers_for(config, model)


@pytest.mark.parametrize(
    "model",
    [
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "moonshotai/Kimi-K2-Instruct",
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
    ],
)
def test_a_checkpoint_that_answers_without_thinking_is_served_without_a_reasoning_parser(
    model,
):
    """Its default reply opens no think block, which vLLM's qwen3 and kimi_k2
    reasoning parsers would serve as reasoning: through auto and through the
    renderer it maps to, it is served with its format's tool parser alone."""
    from renderers import VllmParsers
    from renderers.base import MODEL_RENDERER_MAP, vllm_parsers_for
    from renderers.configs import config_from_name

    config = config_from_name(MODEL_RENDERER_MAP[model])
    tool = vllm_parsers_for(config, "m/fine-tune").tool
    assert vllm_parsers_for(None, model) == VllmParsers(tool, None)
    assert vllm_parsers_for(config, model) == VllmParsers(tool, None)


def test_default_renderer_without_parsers_is_backward_compatible():
    """Without parsers, DefaultRenderer still does basic <think> extraction."""
    from renderers import DefaultRendererConfig, create_renderer

    tok = load_tokenizer("Qwen/Qwen3-0.6B")
    renderer = create_renderer(tok, DefaultRendererConfig())
    assert renderer.supports_tools is False

    ids = tok.encode("<think>r</think>a", add_special_tokens=False)
    parsed = renderer.parse_response(list(ids))
    assert parsed.reasoning_content == "r"
    assert parsed.content == "a"
    assert parsed.tool_calls == []


@pytest.mark.parametrize(
    "text",
    [
        "Before<think>reason</think>After",
        "Before<think>unfinished",
        "\n<think>reason</think>After",
        "Before</think>After",
    ],
)
def test_think_parser_requires_initial_opener(text):
    parser = get_reasoning_parser("think", None)
    assert parser.extract(text) == (None, text)


def test_parse_status_uses_wire_value_as_string():
    for status in ToolCallParseStatus:
        assert str(status) == status.value
        assert f"{status}" == status.value
        assert json.loads(json.dumps({"status": status})) == {"status": status.value}
        assert ToolCallParseStatus(status.value) is status
