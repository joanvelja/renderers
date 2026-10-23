"""Reference template → live generation prefix → client → parsed response.

Only inference HTTP is replayed. Expected tokens come from the checkpoint's
chat template (or independent native encoder), never from our renderer. Every
catalog checkpoint and every declared template-setting combination participates.
"""

import json
from functools import lru_cache
from pathlib import Path

import pytest
from parity import MODEL_CATALOG, TOOLS, kwarg_combinations
from reference_continuations import (
    ANSWER,
    BEFORE,
    BEFORE_KWARGS,
    REASONING,
    SERVED_REPLIES,
    reference_continuation,
    served_key,
    tool_reply_with_text_after,
)
from replay import assert_evidence, replay_client

from renderers import VLLM_CHAT_PARITY, VllmParsers, create_renderer
from renderers.base import (
    MODEL_RENDERER_MAP,
    ToolCallParseStatus,
    load_tokenizer,
    vllm_parsers_for,
)
from renderers.client import generate
from renderers.configs import config_from_name


# Both matrices visit checkpoints in groups; reuse within each group only.
_tokenizer = lru_cache(maxsize=1)(load_tokenizer)
# Default has no configured tool parser; Llama's template permits one call;
# Harmony's handoff ends generation at the first call. Other formats can emit
# multiple calls in one sampled continuation.
_CASES = [
    pytest.param(case, kwargs, calls, id=f"{case.model}-{kwargs}-calls={calls}")
    for case in MODEL_CATALOG
    for kwargs in kwarg_combinations(case)
    for calls in (
        [0]
        if case.resolved_renderer == "default"
        else [0, 1]
        if case.resolved_renderer in {"llama-3", "gpt-oss"}
        else [0, 1, 2]
    )
]


def test_continuation_catalog_covers_registered_renderers():
    assert set(MODEL_RENDERER_MAP.values()) <= {
        case.resolved_renderer for case in MODEL_CATALOG
    }


def _sampled_prefix(renderer, sampled):
    # A server stops at the first configured stop token. Several templates emit
    # trailing newlines after that token, which are history formatting, not sampled.
    stops = renderer.get_stop_token_ids()
    end = next(
        (i + 1 for i, token in enumerate(sampled) if token in stops), len(sampled)
    )
    return sampled[:end]


@pytest.mark.asyncio
@pytest.mark.parametrize("case,kwargs,calls", _CASES)
async def test_reference_continuation_through_generate(case, kwargs, calls):
    tok = _tokenizer(case.model)
    renderer = create_renderer(
        tok,
        config_from_name(case.resolved_renderer),
        chat_template_kwargs=kwargs or None,
    )
    tools = TOOLS if calls else None
    messages = [{"role": "user", "content": "Check the weather."}]
    prompt, sampled, reasoning = reference_continuation(
        tok, case, kwargs, messages, tools, calls=calls
    )
    sampled = _sampled_prefix(renderer, sampled)
    completions = [(sampled, "stop")]
    if reasoning:
        # Cut the SAME reference stream inside thinking, before its close, to
        # exercise length termination with the real generation prefill.
        cut = next(
            i
            for i in range(1, len(sampled) + 1)
            if tok.decode(sampled[:i]).rstrip().endswith(REASONING)
        )
        unfinished = sampled[:cut]
        completions.append((unfinished, "length"))
    requests = []
    async with replay_client(completions, requests) as client:

        async def run():
            return await generate(
                client=client,
                renderer=renderer,
                model=case.model,
                messages=messages,
                tools=tools,
                max_prompt_len=100000,
            )

        result = await run()
        assert requests[0]["token_ids"] == prompt
        assert result["prompt_ids"] == prompt
        assert_evidence(result, sampled)
        assert result["content"].strip() == ("" if calls else ANSWER)
        assert (result["reasoning_content"] or "").strip() == reasoning
        assert result["reasoning_complete"] is (
            None if case.resolved_renderer == "default" else True
        )
        assert result["finish_reason"] == ("tool_calls" if calls else "stop")
        assert len(result["tool_calls"]) == calls
        for call, city in zip(result["tool_calls"], ["Tokyo", "Berlin"]):
            assert call.status == ToolCallParseStatus.OK
            assert call.name == "get_weather"
            arguments = call.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            assert arguments == {"city": city}
        if reasoning:
            truncated = await run()
            assert requests[1]["token_ids"] == prompt
            assert_evidence(truncated, unfinished)
            assert truncated["content"] == ""
            assert truncated["reasoning_content"].strip() == REASONING
            assert truncated["reasoning_complete"] is False
            assert truncated["tool_calls"] == []
            assert truncated["finish_reason"] == "length"


_SERVED = json.loads(
    (Path(__file__).parent / "fixtures" / "vllm_chat_content.json").read_text()
)
_MESSAGES = [{"role": "user", "content": "Check the weather."}]


def _in_renderers_convention(renderer_name: str, content: str) -> str:
    """vLLM's ``content`` under renderers' whitespace convention for the format:
    outer whitespace stripped, as every parse but these does; kept by Inkling's
    and Muse Glimmer's; kept by DeepSeek V4's, whose whitespace-only content is
    "" (``parse_deepseek_v4``)."""
    if renderer_name in {"inkling", "muse-glimmer"}:
        return content
    if renderer_name == "deepseek-v4":
        return content if content.strip() else ""
    return content.strip()


async def _served_without_tool_calls(renderer, model, sampled, kwargs, tools):
    """Parse sampled through generate(tool_calls=False) and check it
    against the content vLLM serves for it (vllm_chat_reference.py)."""
    assert _SERVED["vllm"] == VLLM_CHAT_PARITY
    ctk = {k: v for k, v in kwargs.items() if v is not None}
    key = served_key(model, tools, sampled, ctk)
    assert key in _SERVED["served"], (
        "no vLLM content for this reply: regenerate fixtures/vllm_chat_content.json"
    )
    async with replay_client([(sampled, "stop")], []) as client:
        result = await generate(
            client=client,
            renderer=renderer,
            model=model,
            messages=_MESSAGES,
            tools=tools,
            max_prompt_len=100000,
            tool_calls=False,
        )
    want = _SERVED["contents"][_SERVED["served"][key]]
    # Exactly vLLM's content, up to the format's own whitespace convention on
    # every reply; any other difference is this parse's.
    name = MODEL_RENDERER_MAP[model]
    assert result["content"] == _in_renderers_convention(name, want)
    assert result["tool_calls"] == []
    assert result["finish_reason"] == "stop"
    return result, want


# Every reference tool-call reply, and one per checkpoint with text beside its
# call (``BEFORE_KWARGS``).
_SERVED_CASES = [pytest.param(*p.values, "", id=p.id) for p in _CASES if p.values[2]]
_SERVED_CASES += [
    pytest.param(case, kwargs, 1, BEFORE, id=f"{case.model}-{kwargs}-{BEFORE}-calls=1")
    for case in MODEL_CATALOG
    if case.resolved_renderer != "default"
    and (kwargs := BEFORE_KWARGS.get(case.resolved_renderer, {})) is not None
]


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [TOOLS, None], ids=["tools", "no-tools"])
@pytest.mark.parametrize("case,kwargs,calls,content", _SERVED_CASES)
async def test_reference_tool_calls_parsed_without_tool_calls(
    case, kwargs, calls, content, tools
):
    """``tool_calls=False`` serves a reply carrying tool calls, text after them
    and ``content`` beside them as vLLM's chat path does under
    tool_choice="none" with the pair the checkpoint is served with
    (``vllm_parsers_for``): with the tools, the whole reply; without tools, the
    streamed one (they differ only for an engine-based pair). No calls, finish
    "stop", and the reasoning split of the tool-call parse."""
    tok = _tokenizer(case.model)
    renderer = create_renderer(
        tok,
        config_from_name(case.resolved_renderer),
        chat_template_kwargs=kwargs or None,
    )
    served = vllm_parsers_for(config_from_name(case.resolved_renderer), case.model)
    assert served == VllmParsers(*_SERVED["pairs"][case.model])
    sampled = tool_reply_with_text_after(
        tok, renderer, case, kwargs, _MESSAGES, TOOLS, calls, content
    )
    assert content in tok.decode(sampled)
    _, _, reasoning = reference_continuation(
        tok, case, kwargs, _MESSAGES, TOOLS, calls=calls, content=content
    )
    result, want = await _served_without_tool_calls(
        renderer, case.model, sampled, kwargs, tools
    )
    # The markup rule renderers derives from the served pair is vLLM's.
    assert ("get_weather" in want) is not served.drops_tool_markup
    assert (result["reasoning_content"] or "").strip() == reasoning
    assert result["reasoning_complete"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [TOOLS, None], ids=["tools", "no-tools"])
@pytest.mark.parametrize("reply", sorted(SERVED_REPLIES))
async def test_reply_parsed_without_tool_calls_as_vllm_serves_it(reply, tools):
    """Replies turning on one rule of the format's vLLM parsers
    (``SERVED_REPLIES``), parsed without tool calls."""
    model, text = SERVED_REPLIES[reply]
    tok = _tokenizer(model)
    renderer = create_renderer(tok, config_from_name(MODEL_RENDERER_MAP[model]))
    sampled = tok.encode(text, add_special_tokens=False)
    await _served_without_tool_calls(renderer, model, sampled, {}, tools)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [case for case in MODEL_CATALOG if case.resolved_renderer != "default"],
    ids=lambda case: case.model,
)
async def test_reference_continuation_after_tool_dispatch(case):
    """The tool observation can change the next generation's reasoning prefix."""
    tok = _tokenizer(case.model)
    renderer = create_renderer(tok, config_from_name(case.resolved_renderer))
    messages = [{"role": "user", "content": "Check the weather."}]
    prompt, sampled, _ = reference_continuation(tok, case, {}, messages, TOOLS, calls=1)
    sampled = _sampled_prefix(renderer, sampled)
    requests = []
    async with replay_client([(sampled, "stop")], requests) as client:
        result = await generate(
            client=client,
            renderer=renderer,
            model=case.model,
            messages=messages,
            tools=TOOLS,
            max_prompt_len=100000,
        )
    assert requests[0]["token_ids"] == prompt
    (call,) = result["tool_calls"]
    assert call.status == ToolCallParseStatus.OK
    arguments = (
        json.loads(call.arguments)
        if isinstance(call.arguments, str)
        else call.arguments
    )
    observation = {"get_weather": lambda city: f"Sunny in {city}."}[call.name](
        **arguments
    )
    assistant = {
        "role": "assistant",
        "content": result["content"],
        "tool_calls": [
            {
                "id": "functions.get_weather:0",
                "type": "function",
                "function": {"name": call.name, "arguments": arguments},
            }
        ],
    }
    if result["reasoning_content"]:
        assistant["reasoning_content"] = result["reasoning_content"]
    if case.resolved_renderer in {"deepseek-v3", "deepseek-r1"}:
        assistant["content"] = None
        assistant["tool_calls"][0]["function"]["arguments"] = json.dumps(arguments)
    messages += [
        assistant,
        {
            "role": "tool",
            "name": call.name,
            "tool_call_id": "functions.get_weather:0",
            "content": observation,
        },
    ]
    prompt, sampled, reasoning = reference_continuation(tok, case, {}, messages, TOOLS)
    sampled = _sampled_prefix(renderer, sampled)
    requests = []
    async with replay_client([(sampled, "stop")], requests) as client:
        result = await generate(
            client=client,
            renderer=renderer,
            model=case.model,
            prompt_ids=prompt,
            messages=messages,
            tools=TOOLS,
            max_prompt_len=100000,
        )
    # Feed the exact reference sampling prompt. Historical re-rendering may
    # deliberately retain reasoning that a template drops; it is not the oracle
    # for the parser's live channel state after a tool observation.
    assert requests[0]["token_ids"] == prompt
    assert_evidence(result, sampled)
    assert result["content"].strip() == ANSWER
    assert (result["reasoning_content"] or "").strip() == reasoning
    assert result["reasoning_complete"] is True
    assert result["tool_calls"] == []
    assert result["finish_reason"] == "stop"


def teardown_module():
    """Release model resources when this module finishes, not at process exit."""
    _tokenizer.cache_clear()
