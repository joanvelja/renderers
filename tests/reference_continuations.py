"""Live responses derived from the independent template/encoder oracle.

History serialization is not always a live completion: R1 drops reasoning,
Harmony drops final-turn analysis, and Gemma drops empty thinking prefills.
These explicit adapters describe those reference contracts. Never repair a
prefix mismatch by silently slicing at a common prefix or asking our renderer
to supply the expected output.
"""

import hashlib
import json
import re

from reference_rendering import _harmony_messages, render_reference

REASONING = "Check the weather carefully."
ANSWER = "It is sunny in Tokyo."

MESSAGES = [{"role": "user", "content": "Check the weather."}]
"""The conversation every reply served without tool calls answers
(``served_key``)."""


def reference_continuation(tok, case, kwargs, messages, tools, *, calls=0, content=""):
    """(prompt, completion, reasoning) of the reference's live reply: ``ANSWER``,
    or ``calls`` tool calls with ``content`` beside them."""
    options = {
        k: v for k, v in (dict(case.oracle_defaults) | kwargs).items() if v is not None
    }

    def render(conversation, generation=False, **overrides):
        return render_reference(
            tok,
            conversation,
            oracle=case.oracle,
            tools=tools,
            add_generation_prompt=generation,
            **(options | overrides),
        )

    prompt = render(messages, generation=True)
    prompt_text = tok.decode(prompt, skip_special_tokens=False)
    # Resolve the mode from a reference user-turn prefix even when a tool
    # observation continues the same assistant turn without repeating it.
    first_user = next(
        i for i, message in enumerate(messages) if message["role"] == "user"
    )
    mode_prefix = tok.decode(render(messages[: first_user + 1], generation=True))
    disabled = bool(
        re.search(r"(?:</think>|</think:opensource>|<channel\|>)\s*$", mode_prefix)
    )
    assistant = {"role": "assistant", "content": content if calls else ANSWER}
    if not disabled:
        assistant["reasoning_content"] = REASONING
    if calls:
        assistant["tool_calls"] = [
            {
                "id": f"functions.get_weather:{index}",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": city}},
            }
            for index, city in enumerate(["Tokyo", "Berlin"][:calls])
        ]
        if case.resolved_renderer in {"deepseek-v3", "deepseek-r1"}:
            # These upstream templates require null content and JSON strings.
            assistant["content"] = content or None
            for call in assistant["tool_calls"]:
                fn = call["function"]
                fn["arguments"] = json.dumps(fn["arguments"])

    # Raw-history modes expect an already serialized assistant body. Serialize
    # our structured fixture through the reference's normal branch; these flags
    # do not change the live generation prefix (checked independently below).
    serialization = {
        key: False
        for key in ("raw_last_assistant", "render_assistant_messages_raw")
        if options.get(key)
    }
    assert render(messages, generation=True, **serialization) == prompt
    full = render(messages + [assistant], **serialization)
    full_text = tok.decode(full, skip_special_tokens=False)

    if case.oracle == "harmony" and not calls:
        from openai_harmony import (
            Conversation,
            HarmonyEncodingName,
            Message,
            Role,
            RenderConversationConfig,
            load_harmony_encoding,
        )

        encoder = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        history = _harmony_messages(
            messages,
            tools=tools,
            reasoning_effort=options.get("reasoning_effort", "medium"),
            conversation_start_date=options.get("conversation_start_date"),
        )
        history += [
            Message.from_role_and_content(Role.ASSISTANT, REASONING).with_channel(
                "analysis"
            ),
            Message.from_role_and_content(Role.ASSISTANT, ANSWER).with_channel("final"),
        ]
        full = encoder.render_conversation_for_training(
            Conversation.from_messages(history),
            RenderConversationConfig(auto_drop_analysis=False),
        )
    elif case.resolved_renderer == "deepseek-r1" and prompt_text.endswith("<think>\n"):
        # R1's generation prompt opens thinking, but its history branch removes
        # it. Restore the live reasoning segment before the template's answer.
        prefill = "<think>\n"
        assert prompt_text.endswith(prefill)
        history_prefix = prompt_text[: -len(prefill)]
        assert full_text.startswith(history_prefix)
        full_text = (
            prompt_text + REASONING + "</think>" + full_text[len(history_prefix) :]
        )
        full = tok.encode(full_text, add_special_tokens=False)
    elif case.resolved_renderer == "gemma4" and prompt_text.endswith(
        "<|channel>thought\n<channel|>"
    ):
        prefill = "<|channel>thought\n<channel|>"
        assert prompt_text.endswith(prefill)
        history_prefix = prompt_text[: -len(prefill)]
        assert full_text.startswith(history_prefix)
        full_text = prompt_text + full_text[len(history_prefix) :]
        full = tok.encode(full_text, add_special_tokens=False)
    elif (
        case.resolved_renderer == "laguna-s-2.1"
        and disabled
        and options.get("preserve_thinking")
    ):
        # History preserves an empty opener; generation supplies only the close.
        history_prefix = prompt_text.removesuffix("</think>") + "<think></think>"
        assert full_text.startswith(history_prefix)
        full_text = prompt_text + full_text[len(history_prefix) :]
        full = tok.encode(full_text, add_special_tokens=False)

    if (
        case.resolved_renderer in {"kimi-k2.5", "kimi-k2.6"}
        and messages[-1]["role"] == "tool"
    ):
        # Kimi's final-answer history branch strips the preceding tool turn's
        # reasoning as well as the final turn's reasoning. Extract only the
        # last assistant body and preserve the actual sampling prompt.
        header = "<|im_assistant|>assistant<|im_middle|>"
        body = tok.decode(full).rsplit(header, 1)[1]
        if not disabled:
            assert prompt_text.endswith(header + "<think>")
            assert body.startswith("<think></think>")
            body = REASONING + body[len("<think>") :]
        else:
            assert prompt_text.endswith(header + "</think>")
            assert body.startswith("</think>")
            body = body[len("</think>") :]
        full = tok.encode(prompt_text + body, add_special_tokens=False)

    assert full[: len(prompt)] == prompt, (
        "Reference history is not a live continuation; add an explicit oracle adapter",
        case.model,
        kwargs,
        prompt_text[-200:],
        tok.decode(full)[-300:],
    )
    completion = full[len(prompt) :]
    # Expectations come from the reference, never from parse_response. Some
    # non-reasoning templates intentionally ignore reasoning_content inputs.
    reasoning = REASONING if REASONING in tok.decode(completion) else ""
    return prompt, completion, reasoning


AFTER = "\nAfter."
"""Text a reply carries after its last tool call, where vLLM's tool parsers
part ways once they parse no calls."""

BEFORE = "Before."
"""Content a reference tool-call reply carries beside its call, where the
template puts it (before the call, except Gemma 4 and Harmony's final message)."""

BEFORE_KWARGS: dict[str, dict | None] = {
    # The template can't render text beside a call: DeepSeek V3 drops the
    # calls, Llama 3 and Muse Glimmer the text.
    "deepseek-v3": None,
    "llama-3": None,
    "muse-glimmer": None,
    # The template renders a tool-call turn without its reasoning, while the
    # checkpoints think by default: vLLM's qwen3 parser reads a reply that opens
    # no think block as reasoning, and the template has no switch to turn it off.
    "prime-qwen3": None,
    # vLLM renders DeepSeek V4 in thinking mode unless the request turns it off
    # (vllm/tokenizers/deepseek_v4.py:28-33); renderers defaults to chat mode.
    "deepseek-v4": {"enable_thinking": True},
}
"""Template settings of the reply with ``BEFORE`` beside its call, by renderer
(default: the template's defaults); ``None`` has no such reply."""

# Hand-written replies, each turning on one rule of its format's vLLM parsers:
# name -> (model, completion text).
SERVED_REPLIES = {
    # llama3_json: no parser adjusts the request, so special tokens are skipped.
    "llama-3-python-tag": (
        "meta-llama/Llama-3.2-1B-Instruct",
        '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Tokyo"}}'
        "<|eom_id|>",
    ),
    # muse_glimmer: the to=user body ends before one final LF (`$`).
    "muse-glimmer-final-newline": (
        "meta-models/Muse-Glimmer-30B",
        " to=user<|message|>Hello\n<|eot|>",
    ),
    # muse_glimmer: the whitespace before that LF stays content.
    "muse-glimmer-trailing-whitespace": (
        "meta-models/Muse-Glimmer-30B",
        " to=user<|message|>Sunny in Tokyo.  \n\n<|eot|>",
    ),
    # inkling: a text segment's trailing whitespace stays content.
    "inkling-trailing-whitespace": (
        "thinkingmachines/Inkling",
        "<|content_thinking|>Need weather.<|end_message|><|message_model|>"
        "<|content_text|>Sunny in Tokyo.  \n<|content_model_end_sampling|>",
    ),
    # hermes + qwen3: vLLM keeps the content's trailing whitespace, which
    # renderers strips.
    "qwen3-trailing-whitespace": (
        "Qwen/Qwen3-8B",
        "<think>\nNeed weather.\n</think>\n\nSunny in Tokyo.  \n<|im_end|>",
    ),
    # Harmony: commentary without a recipient is content, the tool message is
    # dropped. The reply follows the analysis header renderers' prompt prefills.
    "gpt-oss-preamble-call": (
        "openai/gpt-oss-20b",
        "Need weather.<|end|><|start|>assistant<|channel|>commentary<|message|>"
        "Let me check.<|end|><|start|>assistant<|channel|>commentary "
        'to=functions.get_weather <|constrain|>json<|message|>{"city": "Tokyo"}'
        "<|call|>",
    ),
    # Harmony: the final channel is content.
    "gpt-oss-final": (
        "openai/gpt-oss-20b",
        "Need weather.<|end|><|start|>assistant<|channel|>final<|message|>"
        "Sunny in Tokyo.<|return|>",
    ),
    # gemma4: text before a call stays content (its template renders text after
    # calls, so no reference reply has it before).
    "gemma4-text-before-call": (
        "google/gemma-4-31B-it",
        'Before.<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}<tool_call|>'
        "<|tool_response>",
    ),
    # qwen3 grammar: a bare <function= opens a call; the text after its
    # </function> stays out of the content until a </tool_call>.
    "qwen3.5-bare-function": (
        "Qwen/Qwen3.5-9B",
        "Check.\n</think>\n\nSure.\n<function=get_weather>\n<parameter=city>\n"
        "Tokyo\n</parameter>\n</function>\nafter<|im_end|>",
    ),
}


def tool_reply_with_text_after(
    tok, renderer, case, kwargs, messages, tools, calls, content=""
):
    """The reference tool-call continuation (``content`` beside the calls) up to
    the renderer's first stop token, with ``AFTER`` inserted before that stop."""
    _, sampled, _ = reference_continuation(
        tok, case, kwargs, messages, tools, calls=calls, content=content
    )
    stops = set(renderer.get_stop_token_ids())
    end = next((i for i, token in enumerate(sampled) if token in stops), None)
    after = tok.encode(AFTER, add_special_tokens=False)
    if end is None:
        return sampled + after
    return sampled[:end] + after + [sampled[end]]


def reasoning_left_in_content(model, chat_template_kwargs):
    """Whether the pair ``model`` is served with leaves a reply's reasoning in
    vLLM's content, where renderers splits it off, for a request sending
    ``chat_template_kwargs``: the README's listed gaps, by checkpoint. There the
    fixture pins the content after vLLM's reasoning closer and the split is
    renderers' own; anywhere else reasoning left in the content is the pair's."""
    kwargs = chat_template_kwargs
    if model == "tencent/Hy3":
        # hy_v3 doesn't split under reasoning_effort "no_think", its default
        # (vllm/reasoning/hy_v3_reasoning_parser.py:40-50).
        return kwargs.get("reasoning_effort", "no_think") == "no_think"
    if model == "poolside/Laguna-S-2.1":
        # poolside_v1 splits only when the request turns thinking on
        # (deepseek_v3_reasoning_parser.py:28-37); the template thinks unasked.
        return not (kwargs.get("thinking") or kwargs.get("enable_thinking"))
    if model in {"Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"}:
        # Served without a reasoning parser (MODEL_VLLM_PARSERS).
        return kwargs.get("enable_thinking") is True
    # minimax_m2_append_think serves the whole reply as content, behind a
    # "<think>" (minimax_m2_reasoning_parser.py:58-61).
    return model == "MiniMaxAI/MiniMax-M2.5"


def served_key(model, tools, completion_ids, chat_template_kwargs):
    """The fixture's key for a reply: a digest of the model, whether the request
    offers tools, the reply's token ids and the template kwargs it sends. It
    names what vLLM serves only for the fixed ``MESSAGES`` and ``TOOLS``. The
    prompt isn't in it: gpt-oss's and Muse Glimmer's print today's date, which
    vLLM's parse doesn't read, so a key over it would expire the fixture daily."""
    inputs = [model, bool(tools), completion_ids, chat_template_kwargs]
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]
