from __future__ import annotations

from functools import lru_cache

import pytest

from renderers import (
    Gemma4Renderer,
    Gemma4RendererConfig,
    config_from_name,
    create_renderer,
)
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
                    "city": {"type": "string", "description": "The city name"},
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

RESPONSE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "Summarize text",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "response": {"type": "string", "description": "summary text"},
        },
    }
]

MULTI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "first",
            "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "second",
            "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []},
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
        messages,
        tokenize=True,
        return_dict=False,
        **kwargs,
    )
    if isinstance(result, dict):
        return list(result["input_ids"])
    return list(result)


def test_gemma4_registered_as_typed_custom_renderer():
    assert config_from_name("gemma4") == Gemma4RendererConfig()
    assert Gemma4RendererConfig.template_field_names() == frozenset({"enable_thinking"})
    assert {m for m, r in MODEL_RENDERER_MAP.items() if r == "gemma4"} == GEMMA4_MODELS

    _, renderer = _gemma4()
    assert type(renderer).__name__ == "Gemma4Renderer"


def test_gemma4_rejects_empty_messages():
    _, renderer = _gemma4()

    with pytest.raises(ValueError, match="No messages provided"):
        renderer.render([])


@pytest.mark.parametrize(
    ("gemma4_model_name", "expected"),
    [
        ("google/gemma-4-E2B", False),
        ("google/gemma-4-E4B", False),
        ("google/gemma-4-31B", True),
        ("google/gemma-4-26B-A4B", True),
    ],
)
def test_gemma4_base_checkpoint_prompt_variant_fallback_is_exact(
    gemma4_model_name,
    expected,
):
    class StubTokenizer:
        name_or_path = gemma4_model_name
        chat_template = None
        unk_token_id = -1

        def convert_tokens_to_ids(self, token):
            return {
                "<turn|>": 106,
                "<|tool_call>": 48,
                "<tool_call|>": 49,
                "<|tool_response>": 50,
                "<eos>": 1,
            }[token]

    renderer = Gemma4Renderer(StubTokenizer())

    assert renderer._add_empty_thought_generation_prompt is expected


@pytest.mark.parametrize(
    "gemma4_model_name",
    [
        "google/gemma-4-E2B-it",
        "google/gemma-4-E4B-it",
        "google/gemma-4-31B-it",
        "google/gemma-4-26B-A4B-it",
    ],
)
@pytest.mark.parametrize("enable_thinking", [True, False])
def test_gemma4_generation_prompt_parity_across_it_sizes(
    gemma4_model_name,
    enable_thinking,
):
    tokenizer = load_tokenizer(gemma4_model_name, use_fastokens=False)
    renderer = create_renderer(
        tokenizer,
        Gemma4RendererConfig(enable_thinking=enable_thinking),
    )
    messages = [{"role": "user", "content": "Hello!"}]

    assert renderer.render_ids(messages, add_generation_prompt=True) == _expected(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_gemma4_text_and_tool_chat_parity_with_hf_template(enable_thinking):
    tokenizer = load_tokenizer("google/gemma-4-E2B-it", use_fastokens=False)
    renderer = create_renderer(
        tokenizer, Gemma4RendererConfig(enable_thinking=enable_thinking)
    )
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
        ([{"role": "user", "content": "Weather?"}], {"tools": TOOLS}),
        ([{"role": "user", "content": "Summarize."}], {"tools": RESPONSE_TOOL}),
        (
            [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "reasoning_content": "Need weather.",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Paris", "days": None},
                            }
                        }
                    ],
                },
            ],
            {"tools": TOOLS},
        ),
        (
            [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Paris"},
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 20}'},
                {"role": "assistant", "content": "It is 20 degrees."},
            ],
            {"tools": TOOLS, "add_generation_prompt": True},
        ),
        (
            [
                {"role": "user", "content": "Call both."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "first", "arguments": {}}},
                        {"function": {"name": "second", "arguments": {}}},
                    ],
                },
                {"role": "tool", "content": "first result"},
                {"role": "tool", "content": "second result"},
                {"role": "assistant", "content": "done"},
            ],
            {"tools": MULTI_TOOLS},
        ),
    ]

    for messages, kwargs in cases:
        expected_kwargs = dict(kwargs)
        expected_kwargs["enable_thinking"] = enable_thinking
        assert renderer.render_ids(messages, **kwargs) == _expected(
            tokenizer,
            messages,
            **expected_kwargs,
        )


def test_gemma4_tool_metadata_and_masks():
    _, renderer = _gemma4()
    messages = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 20}'},
        {"role": "assistant", "content": "It is 20 degrees."},
    ]

    rendered = renderer.render(messages, tools=TOOLS)

    assert len(rendered.token_ids) == len(rendered.message_indices)
    assert len(rendered.token_ids) == len(rendered.sampled_mask)
    assert len(rendered.token_ids) == len(rendered.is_content)
    assert rendered.message_tool_names == [None, None, "get_weather", None]
    assert any(
        idx == 2 and is_content and not sampled
        for idx, is_content, sampled in zip(
            rendered.message_indices,
            rendered.is_content,
            rendered.sampled_mask,
        )
    )
    assert not any(
        idx == 2 and sampled
        for idx, sampled in zip(rendered.message_indices, rendered.sampled_mask)
    )


def test_gemma4_parse_tool_calls_and_reasoning():
    tokenizer, renderer = _gemma4()
    text = (
        "<|channel>thought\nNeed weather.\n<channel|>"
        "Done"
        '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>,'
        "meta:{ok:true,temps:[1,2.5,None]}}<tool_call|>"
        '<|tool_call>call:get_time{zone:<|"|>UTC<|"|>}<tool_call|>'
        '<|tool_call>call:get_weather{"city":"Rome","meta":{"ok":false}}'
        "<tool_call|>"
        "<|tool_response>"
    )
    ids = tokenizer.encode(text, add_special_tokens=False)

    parsed = renderer.parse_response(ids, tools=TOOLS)

    assert parsed.content == "Done"
    assert parsed.reasoning_content == "Need weather."
    assert [call.status for call in parsed.tool_calls] == [
        ToolCallParseStatus.OK,
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
    assert parsed.tool_calls[2].name == "get_weather"
    assert parsed.tool_calls[2].arguments == {
        "city": "Rome",
        "meta": {"ok": False},
    }


def test_gemma4_render_accepts_openai_json_string_arguments():
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
        messages[:1],
        tools=TOOLS,
        add_generation_prompt=True,
    )
    completion_ids = renderer.render_ids(messages, tools=TOOLS)[len(prompt_ids) :]
    parsed = renderer.parse_response(completion_ids, tools=TOOLS)

    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].status == ToolCallParseStatus.OK
    assert parsed.tool_calls[0].arguments == {"city": "Tokyo", "days": 2}


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_gemma4_bridge_matches_full_render(enable_thinking):
    tokenizer = load_tokenizer("google/gemma-4-E2B-it", use_fastokens=False)
    renderer = create_renderer(
        tokenizer, Gemma4RendererConfig(enable_thinking=enable_thinking)
    )
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
    assert bridged.message_tool_names == [None]


def test_gemma4_rejects_non_text_multimodal_parts_until_sidecar_exists():
    _, renderer = _gemma4()

    with pytest.raises(ValueError, match="multimodal sidecar"):
        renderer.render(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this."},
                        {"type": "image", "image": object()},
                    ],
                }
            ]
        )
