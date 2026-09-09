"""Focused coverage for Gemma 4's template variants and tool grammar."""

from functools import lru_cache

import pytest

from renderers import Gemma4Renderer, create_renderer
from renderers.base import MODEL_RENDERER_MAP, MULTIMODAL_MODELS, load_tokenizer
from renderers.configs import Gemma4RendererConfig


_MODELS = {
    "google/gemma-4-E2B",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B",
    "google/gemma-4-E4B-it",
    "google/gemma-4-12B",
    "google/gemma-4-12B-it",
    "google/gemma-4-26B-A4B",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B",
    "google/gemma-4-31B-it",
}


@lru_cache
def _gemma4():
    tokenizer = load_tokenizer("google/gemma-4-31B-it")
    return tokenizer, create_renderer(tokenizer)


def test_all_checkpoints_are_registered_as_image_renderers():
    for model in _MODELS:
        assert MODEL_RENDERER_MAP[model] == "gemma4"
        assert MULTIMODAL_MODELS[model] == {"image"}


def test_disabled_thinking_prefill_tracks_template_revision(monkeypatch):
    tokenizer, current_renderer = _gemma4()
    messages = [{"role": "user", "content": "Hello"}]

    current_text = tokenizer.decode(
        current_renderer.render_ids(messages, add_generation_prompt=True),
        skip_special_tokens=False,
    )
    assert current_text.endswith("<|channel>thought\n<channel|>")

    # E2B/E4B use the otherwise-identical earlier template revision, which
    # stops at the model role opener when thinking is disabled.
    monkeypatch.setattr(tokenizer, "name_or_path", "google/gemma-4-E4B-it")
    monkeypatch.setattr(tokenizer, "chat_template", "")
    earlier_renderer = Gemma4Renderer(tokenizer)
    earlier_text = tokenizer.decode(
        earlier_renderer.render_ids(messages, add_generation_prompt=True),
        skip_special_tokens=False,
    )
    assert earlier_text.endswith("<|turn>model\n")


@pytest.mark.parametrize(
    ("gemma4_model_name", "has_empty_thought"),
    [
        ("google/gemma-4-E2B", False),
        ("google/gemma-4-E4B", False),
        ("google/gemma-4-12B", True),
        ("google/gemma-4-26B-A4B", True),
        ("google/gemma-4-31B", True),
    ],
)
def test_base_checkpoint_prompt_variant_fallback_is_exact(
    monkeypatch, gemma4_model_name, has_empty_thought
):
    tokenizer, _ = _gemma4()
    monkeypatch.setattr(tokenizer, "name_or_path", gemma4_model_name)
    monkeypatch.setattr(tokenizer, "chat_template", "")
    renderer = Gemma4Renderer(tokenizer)

    text = tokenizer.decode(
        renderer.render_ids(
            [{"role": "user", "content": "Hello"}],
            add_generation_prompt=True,
        ),
        skip_special_tokens=False,
    )

    assert text.endswith("<|channel>thought\n<channel|>") is has_empty_thought


def test_enable_thinking_controls_derived_retention_and_rejects_conflicts():
    tokenizer, _ = _gemma4()
    disabled = Gemma4Renderer(tokenizer, Gemma4RendererConfig(enable_thinking=False))
    assert disabled.effective_thinking_retention == "all"

    enabled = Gemma4Renderer(tokenizer, Gemma4RendererConfig(enable_thinking=True))
    assert enabled.effective_thinking_retention == "tool_cycle"

    # preserve_thinking widens only the tool-call gate; the bridge policy follows enable_thinking.
    preserved = Gemma4Renderer(
        tokenizer,
        Gemma4RendererConfig(enable_thinking=True, preserve_thinking=True),
    )
    assert preserved.effective_thinking_retention == "tool_cycle"

    # Narrowing the bridge policy is always safe; widening it with thinking on is not.
    conservative = Gemma4Renderer(
        tokenizer,
        Gemma4RendererConfig(enable_thinking=False, thinking_retention="tool_cycle"),
    )
    assert conservative.effective_thinking_retention == "tool_cycle"
    with pytest.raises(ValueError, match="enable_thinking=True implies"):
        Gemma4RendererConfig(enable_thinking=True, thinking_retention="all")


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "tool", "content": "orphan"}],
        [
            {"role": "assistant", "content": "No call."},
            {"role": "tool", "content": "orphan"},
        ],
        [
            {
                "role": "assistant",
                "content": "",
                "tool_responses": [{"name": "legacy", "response": "done"}],
            },
            {"role": "tool", "content": "still orphaned"},
        ],
    ],
)
def test_unconsumed_tool_messages_raise(messages):
    tokenizer, renderer = _gemma4()
    with pytest.raises(ValueError, match="Unconsumed tool message"):
        renderer.render_ids(messages)


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_tool_cycle_matches_canonical_template(enable_thinking):
    tokenizer, _ = _gemma4()
    renderer = Gemma4Renderer(
        tokenizer, Gemma4RendererConfig(enable_thinking=enable_thinking)
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Look up the weather.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [
        {"role": "user", "content": "Weather in Berlin?"},
        {
            "role": "assistant",
            "reasoning_content": "I should call the weather tool."
            if not enable_thinking
            else None,
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": {"city": "Berlin"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"temperature": 24, "unit": "C"}',
        },
        {
            "role": "assistant",
            "reasoning_content": "I can now answer." if not enable_thinking else None,
            "content": "It is 24 C.",
        },
    ]

    expected = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
        return_dict=False,
    )
    assert renderer.render_ids(messages, tools=tools) == list(expected)


def test_disabled_thinking_post_tool_completion_matches_sampled_stream():
    """A post-tool assistant message continues the existing model turn.

    The 12B/26B/31B generation prompt prefills an empty thought channel before the
    initial completion, but the disabled-thinking post-tool prompt does not
    insert another one. A later full rerender must preserve that exact stream.
    """
    tokenizer, _ = _gemma4()
    renderer = Gemma4Renderer(
        tokenizer,
        Gemma4RendererConfig(
            enable_thinking=False,
            preserve_thinking=True,
        ),
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Look up the weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    user = {"role": "user", "content": "Weather in Berlin?"}
    tool_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": {"city": "Berlin"},
                },
            }
        ],
    }
    tool_response = {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"temperature": 24}',
    }
    final = {"role": "assistant", "content": "It is 24 C."}

    initial_prompt = renderer.render_ids(
        [user], tools=tools, add_generation_prompt=True
    )
    tool_call_prompt = renderer.render_ids(
        [user, tool_call], tools=tools, add_generation_prompt=True
    )
    assert tool_call_prompt[: len(initial_prompt)] == initial_prompt
    tool_call_completion = tool_call_prompt[len(initial_prompt) :]

    post_tool_prompt = renderer.bridge_to_next_turn(
        initial_prompt,
        tool_call_completion,
        [tool_response],
        tools=tools,
    )
    assert post_tool_prompt is not None

    final_completion = tokenizer.encode(final["content"], add_special_tokens=False) + [
        renderer.get_stop_token_ids()[0]
    ]
    reminder = {"role": "user", "content": "Please summarize."}
    extended_stream = renderer.bridge_to_next_turn(
        post_tool_prompt.token_ids,
        final_completion,
        [reminder],
        tools=tools,
    )
    assert extended_stream is not None
    rerendered = renderer.render_ids(
        [user, tool_call, tool_response, final, reminder],
        tools=tools,
        add_generation_prompt=True,
    )
    assert rerendered == extended_stream.token_ids


def test_parser_extracts_reasoning_and_multiple_typed_tool_calls():
    tokenizer, renderer = _gemma4()
    text = (
        "<|channel>thought\nI need two lookups.\n<channel|>"
        '<|tool_call>call:weather{city:<|"|>Berlin<|"|>,days:2}'
        "<tool_call|>"
        "<|tool_call>call:flags{enabled:true,values:[1,null]}<tool_call|>"
    )
    parsed = renderer.parse_response(tokenizer.encode(text, add_special_tokens=False))

    assert parsed.reasoning_content == "I need two lookups."
    assert parsed.content == ""
    assert [(call.name, call.arguments) for call in parsed.tool_calls] == [
        ("weather", {"city": "Berlin", "days": 2}),
        ("flags", {"enabled": True, "values": [1, None]}),
    ]


def test_parser_recovers_prompt_opened_post_tool_reasoning():
    tokenizer, _ = _gemma4()
    renderer = Gemma4Renderer(tokenizer, Gemma4RendererConfig(enable_thinking=True))
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "weather", "arguments": {"city": "Berlin"}},
    }
    messages = [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
    ]
    expected_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=False,
    )
    prompt = renderer.render_ids(messages, add_generation_prompt=True)

    assert prompt == list(expected_prompt)
    assert tokenizer.decode(prompt, skip_special_tokens=False).endswith(
        "<|channel>thought\n"
    )

    completion = tokenizer.encode(
        "Need synthesize.\n<channel|>It is sunny.<turn|>",
        add_special_tokens=False,
    )

    parsed = renderer.parse_response(completion)

    assert parsed.reasoning_content == "Need synthesize."
    assert parsed.content == "It is sunny."
    assert parsed.tool_calls == []

    # Initial-turn content without a channel closer remains ordinary content.
    direct = renderer.parse_response(
        tokenizer.encode("Direct answer.<turn|>", add_special_tokens=False)
    )
    assert direct.reasoning_content is None
    assert direct.content == "Direct answer."


def _image_renderer():
    """A Gemma 4 renderer with a live ``Gemma4Processor``, or a skip."""
    tokenizer, _ = _gemma4()
    renderer = Gemma4Renderer(tokenizer)
    try:
        renderer._get_processor()
    except (RuntimeError, OSError) as exc:  # pragma: no cover - env dependent
        pytest.skip(f"Gemma4Processor unavailable: {exc}")
    return tokenizer, renderer


def _tiny_image():
    from PIL import Image

    return Image.new("RGB", (224, 224), color=(128, 192, 255))


@pytest.mark.parametrize("size", [(224, 224), (448, 224), (224, 448)])
def test_real_processor_keeps_one_batched_row_per_image(size):
    """Guard the ``MultiModalFieldConfig.batched('image')`` contract with
    live Gemma4Processor output rather than synthetic tensor shapes."""
    from PIL import Image

    _, renderer = _image_renderer()
    image = Image.new("RGB", size, color=(128, 192, 255))
    rendered = renderer.render(
        [{"role": "user", "content": [{"type": "image", "image": image}]}]
    )
    item = rendered.multi_modal_data.mm_items["image"][0]
    placeholder = rendered.multi_modal_data.mm_placeholders["image"][0]

    assert item["pixel_values"].shape[0] == 1
    assert item["image_position_ids"].shape[0] == 1
    assert item["pixel_values"].shape[1] == item["image_position_ids"].shape[1]
    assert placeholder.length > 0


def test_schema_unified_image_parts_still_expand_image_tokens():
    """``Dataset.from_list`` unifies the Arrow schema across a content list,
    so an image part round-tripped through a dataset carries ``text: None``
    (and text parts carry ``image: None``). Dispatching on ``"text" in part``
    before classifying media would route the image into the text branch,
    emit nothing, and silently skip soft-token expansion."""
    tokenizer, renderer = _image_renderer()
    image = _tiny_image()
    plain = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is this?"},
            ],
        }
    ]
    schema_unified = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "text": None},
                {"type": "text", "text": "What is this?", "image": None},
            ],
        }
    ]

    baseline = renderer.render(plain)
    roundtripped = renderer.render(schema_unified)

    image_id = tokenizer.convert_tokens_to_ids("<|image|>")
    assert baseline.token_ids.count(image_id) > 0
    assert roundtripped.token_ids == baseline.token_ids
    assert (
        roundtripped.multi_modal_data.mm_hashes == baseline.multi_modal_data.mm_hashes
    )
    assert (
        roundtripped.multi_modal_data.mm_placeholders
        == baseline.multi_modal_data.mm_placeholders
    )


def test_schema_unified_tool_response_image_parts_survive():
    """Same hazard on the tool-response path, which also has to keep
    accepting untyped text parts."""
    tokenizer, renderer = _image_renderer()
    image = _tiny_image()
    messages = [
        {"role": "user", "content": "Look it up."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "screenshot", "arguments": {}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": [
                {"type": "image", "image": image, "text": None},
                {"type": "text", "text": "captured", "image": None},
            ],
        },
    ]
    rendered = renderer.render(messages)

    image_id = tokenizer.convert_tokens_to_ids("<|image|>")
    assert rendered.token_ids.count(image_id) > 0
    assert rendered.multi_modal_data.mm_hashes["image"]
    assert "captured" in tokenizer.decode(rendered.token_ids, skip_special_tokens=False)


def test_untyped_text_parts_render_in_tool_responses():
    tokenizer, _ = _gemma4()
    renderer = Gemma4Renderer(tokenizer)
    messages = [
        {"role": "user", "content": "Check."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "check", "arguments": {}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": [{"text": "all good"}],
        },
    ]
    text = tokenizer.decode(renderer.render_ids(messages), skip_special_tokens=False)
    assert "all good" in text


def test_system_content_lists_reject_media_parts():
    """The text-only guard must not be fooled by a schema-unified image
    part's ``text: None`` key, which would drop the image silently."""
    tokenizer, _ = _gemma4()
    renderer = Gemma4Renderer(tokenizer)
    messages = [
        {
            "role": "system",
            "content": [{"type": "image", "image": object(), "text": None}],
        },
        {"role": "user", "content": "Hi"},
    ]
    with pytest.raises(ValueError, match="text parts only"):
        renderer.render_ids(messages)


def test_legacy_assistant_tool_responses_preserve_mask_contract():
    tokenizer, _ = _gemma4()
    renderer = Gemma4Renderer(tokenizer)
    messages = [
        {"role": "user", "content": "Check."},
        {
            "role": "assistant",
            "content": "Done.",
            "tool_responses": [{"name": "check", "response": {"ok": True}}],
        },
    ]
    rendered = renderer.render(messages)

    for index, message_index in enumerate(rendered.message_indices):
        if message_index == 1:
            assert rendered.is_content[index] == rendered.sampled_mask[index]


def test_unnamed_tool_responses_use_ordered_call_names():
    tokenizer, renderer = _gemma4()
    messages = [
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
    ]

    rendered = tokenizer.decode(
        renderer.render_ids(messages),
        skip_special_tokens=False,
    )

    assert "<|tool_response>response:first{" in rendered
    assert "<|tool_response>response:second{" in rendered


def test_render_accepts_openai_json_string_arguments():
    tokenizer, renderer = _gemma4()
    tool = {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Look up weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
    messages = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "weather",
                        "arguments": '{"city": "Tokyo"}',
                    }
                }
            ],
        },
    ]

    prompt_ids = renderer.render_ids(
        messages[:1], tools=[tool], add_generation_prompt=True
    )
    completion_ids = renderer.render_ids(messages, tools=[tool])[len(prompt_ids) :]
    parsed = renderer.parse_response(completion_ids, tools=[tool])

    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].arguments == {"city": "Tokyo"}


def test_bridge_across_user_turn_is_exact_with_thinking_off():
    tokenizer, _ = _gemma4()
    renderer = Gemma4Renderer(tokenizer, Gemma4RendererConfig(enable_thinking=False))
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


@pytest.mark.parametrize("preserve_thinking", [False, True])
def test_bridge_drops_thinking_at_new_user_turn(preserve_thinking):
    """With thinking on, the template strips a non-tool-call turn's reasoning once a
    later user query exists, so the sampled stream cannot be extended in place."""
    tokenizer, _ = _gemma4()
    renderer = Gemma4Renderer(
        tokenizer,
        Gemma4RendererConfig(enable_thinking=True, preserve_thinking=preserve_thinking),
    )
    first = [{"role": "user", "content": "A"}]
    assistant = {"role": "assistant", "content": "B", "reasoning_content": "why B"}
    next_turn = [{"role": "user", "content": "C"}]

    previous_prompt_ids = renderer.render_ids(first, add_generation_prompt=True)
    previous_full_ids = renderer.render_ids(first + [assistant])
    previous_completion_ids = previous_full_ids[len(previous_prompt_ids) :]
    assert "why B" in tokenizer.decode(previous_completion_ids)

    rerendered = renderer.render_ids(
        first + [assistant] + next_turn, add_generation_prompt=True
    )
    assert "why B" not in tokenizer.decode(rerendered)
    assert (
        renderer.bridge_to_next_turn(
            previous_prompt_ids,
            previous_completion_ids,
            next_turn,
        )
        is None
    )


@pytest.mark.parametrize(
    ("messages", "match"),
    [
        ([{"role": "critic", "content": "No."}], "unsupported role"),
        ([{"role": "user"}], "missing content"),
        ([{"role": "user", "content": 7}], "content must be"),
        (
            [{"role": "assistant", "content": "", "reasoning_content": 7}],
            "reasoning_content must be a string",
        ),
        (
            [{"role": "user", "content": [{"text": "missing type"}]}],
            "content part type",
        ),
        (
            [
                {
                    "role": "user",
                    "content": [{"type": "citation", "text": "x"}],
                }
            ],
            "unsupported content part type",
        ),
        (
            [{"role": "user", "content": [{"type": "text", "text": 7}]}],
            "text must be a string",
        ),
        (
            [{"role": "user", "content": [object()]}],
            "content part 0 must be a string or mapping",
        ),
        (
            [{"role": "assistant", "content": "", "tool_calls": ["bad"]}],
            "tool call must be a mapping",
        ),
        (
            [{"role": "assistant", "content": "", "tool_calls": [{}]}],
            "function must be a mapping",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "", "arguments": {}}}],
                }
            ],
            "function name",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "f", "arguments": []}}],
                }
            ],
            "arguments.*JSON object",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_responses": [{"response": "ok"}],
                }
            ],
            "response name",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_responses": [{"name": "f"}],
                }
            ],
            "missing response",
        ),
        ([{"role": "tool", "content": "orphan"}], "Unconsumed tool message"),
        (
            [
                {"role": "user", "content": "call"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "f", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_2", "content": "bad"},
            ],
            "does not match",
        ),
        (
            [
                {"role": "user", "content": "call twice"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "f", "arguments": {}}},
                        {"function": {"name": "f", "arguments": {}}},
                    ],
                },
                {"role": "tool", "content": "first"},
                {"role": "tool", "content": "second"},
                {"role": "tool", "content": "extra"},
            ],
            "has no issuing call at position 2",
        ),
        (
            [
                {"role": "user", "content": "call"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "actual", "arguments": {}}}],
                },
                {"role": "tool", "name": "different", "content": "wrong"},
            ],
            "does not match an issuing call",
        ),
    ],
)
def test_rejects_malformed_messages(messages, match):
    _, renderer = _gemma4()

    with pytest.raises(ValueError, match=match):
        renderer.render(messages)
