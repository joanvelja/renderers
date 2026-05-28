from __future__ import annotations

from functools import lru_cache

from renderers import Gemma4RendererConfig, config_from_name, create_renderer
from renderers.base import MODEL_RENDERER_MAP, ToolCallParseStatus, load_tokenizer

GEMMA4_MODELS = {
    "google/gemma-4-E2B",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B",
    "google/gemma-4-E4B-it",
    "google/gemma-4-31B",
    "google/gemma-4-31B-it",
    "google/gemma-4-26B-A4B",
    "google/gemma-4-26B-A4B-it",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time for a zone",
            "parameters": {
                "type": "object",
                "properties": {"zone": {"type": "string"}},
                "required": ["zone"],
            },
        },
    },
]


@lru_cache(maxsize=1)
def _gemma4():
    tokenizer = load_tokenizer("google/gemma-4-E2B-it", use_fastokens=False)
    renderer = create_renderer(tokenizer)
    return tokenizer, renderer


def _expected(tokenizer, messages, **kwargs) -> list[int]:
    kwargs.setdefault("add_generation_prompt", False)
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, **kwargs
    )
    if isinstance(result, dict):
        return list(result["input_ids"])
    return list(result)


def test_gemma4_registered_as_typed_custom_renderer():
    assert config_from_name("gemma4") == Gemma4RendererConfig()
    assert {m for m, r in MODEL_RENDERER_MAP.items() if r == "gemma4"} == GEMMA4_MODELS

    _, renderer = _gemma4()
    assert type(renderer).__name__ == "Gemma4Renderer"


def test_gemma4_text_chat_parity_with_hf_template():
    tokenizer, renderer = _gemma4()
    cases = [
        ([{"role": "user", "content": "Hello!"}], {"add_generation_prompt": True}),
        (
            [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            {},
        ),
        (
            [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
                {"role": "user", "content": "C"},
            ],
            {"add_generation_prompt": True},
        ),
    ]

    for messages, kwargs in cases:
        assert renderer.render_ids(messages, **kwargs) == _expected(
            tokenizer, messages, **kwargs
        )


def test_gemma4_text_chat_masks_content_and_sampled_assistant_tokens():
    _, renderer = _gemma4()
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]

    rendered = renderer.render(messages)

    assert len(rendered.token_ids) == len(rendered.message_indices)
    assert len(rendered.token_ids) == len(rendered.sampled_mask)
    assert len(rendered.token_ids) == len(rendered.is_content)
    assert rendered.message_roles == ["user", "assistant"]
    assert any(
        msg_idx == 0 and is_content and not sampled
        for msg_idx, is_content, sampled in zip(
            rendered.message_indices, rendered.is_content, rendered.sampled_mask
        )
    )
    assert any(
        msg_idx == 1 and is_content and sampled
        for msg_idx, is_content, sampled in zip(
            rendered.message_indices, rendered.is_content, rendered.sampled_mask
        )
    )


def test_gemma4_tool_template_parity_and_parse_response():
    tokenizer, renderer = _gemma4()
    messages = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris", "unit": "C", "days": 3},
                    }
                }
            ],
        },
    ]

    assert renderer.render_ids(messages, tools=TOOLS) == _expected(
        tokenizer, messages, tools=TOOLS
    )

    prompt_ids = renderer.render_ids(
        messages[:1], tools=TOOLS, add_generation_prompt=True
    )
    completion_ids = renderer.render_ids(messages, tools=TOOLS)[len(prompt_ids) :]
    parsed = renderer.parse_response(completion_ids, tools=TOOLS)

    assert parsed.content == "Let me check."
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.status == ToolCallParseStatus.OK
    assert call.name == "get_weather"
    assert call.arguments == {"city": "Paris", "unit": "C", "days": 3}


def test_gemma4_tool_render_accepts_openai_json_string_arguments():
    _, renderer = _gemma4()
    messages = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Tokyo", "days": 2}',
                    }
                }
            ],
        },
    ]

    prompt_ids = renderer.render_ids(
        messages[:1], tools=TOOLS, add_generation_prompt=True
    )
    completion_ids = renderer.render_ids(messages, tools=TOOLS)[len(prompt_ids) :]
    parsed = renderer.parse_response(completion_ids, tools=TOOLS)

    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].status == ToolCallParseStatus.OK
    assert parsed.tool_calls[0].arguments == {"city": "Tokyo", "days": 2}


def test_gemma4_parse_multiple_tool_calls_and_nested_arguments():
    tokenizer, renderer = _gemma4()
    text = (
        "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,"
        "meta:{ok:true,temps:[1,2.5,null]}}<tool_call|>"
        "<|tool_call>call:get_time{zone:<|\"|>UTC<|\"|>}<tool_call|>"
        "<|tool_response>"
    )
    ids = tokenizer.encode(text, add_special_tokens=False)

    parsed = renderer.parse_response(ids, tools=TOOLS)

    assert parsed.content == ""
    assert [call.status for call in parsed.tool_calls] == [
        ToolCallParseStatus.OK,
        ToolCallParseStatus.OK,
    ]
    assert parsed.tool_calls[0].name == "get_weather"
    assert parsed.tool_calls[0].arguments == {
        "city": "Paris",
        "meta": {"ok": True, "temps": [1, 2.5, None]},
    }
    assert parsed.tool_calls[1].name == "get_time"
    assert parsed.tool_calls[1].arguments == {"zone": "UTC"}


def test_gemma4_bridge_matches_full_render():
    _, renderer = _gemma4()
    first = [{"role": "user", "content": "A"}]
    assistant = {"role": "assistant", "content": "B"}
    next_turn = [{"role": "user", "content": "C"}]

    previous_prompt_ids = renderer.render_ids(first, add_generation_prompt=True)
    previous_full_ids = renderer.render_ids(first + [assistant])
    previous_completion_ids = previous_full_ids[len(previous_prompt_ids) :]
    bridged = renderer.bridge_to_next_turn(
        previous_prompt_ids,
        previous_completion_ids,
        next_turn,
    )

    assert bridged is not None
    assert bridged.token_ids == renderer.render_ids(
        first + [assistant] + next_turn,
        add_generation_prompt=True,
    )
