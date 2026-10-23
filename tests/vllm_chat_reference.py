"""Regenerate ``fixtures/vllm_chat_content.json``: the content and reasoning
vLLM's chat path serves for each reply ``test_roundtrip.py`` parses without tool
calls.

Two steps, from the repo root. The replies come from this project's environment,
so they are the ones the tests build:
``uv run python tests/vllm_chat_reference.py replies replies.json``. vLLM then
serves them where ``VLLM_CHAT_PARITY`` imports (Linux):
``uv run --no-project --with vllm==0.32.0 --with-editable . python
tests/vllm_chat_reference.py serve replies.json``.

Each reply goes through vLLM's own parser stack as ``chat_completion/serving.py``
drives it, with the pair its model is served with (``vllm_parsers_for``) and
``tool_choice="none"``: with the tools, the whole reply (``:997-1047``); without
tools, the streamed reply for an engine-based pair (``:666``) and the whole one
otherwise. Its output is detokenized by vLLM's own v1 detokenizer
(``IncrementalDetokenizer``, primed with the prompt, the stop token excluded),
fast or slow as the tokenizer is. A Harmony
(gpt-oss) reply is parsed from the channel header renderers' prompt prefills,
where vLLM's prompt ends.

The fixture maps a digest of each reply's inputs (``served_key``: the model,
whether tools are offered, the reply's ids and the template kwargs) to an index
into its ``messages``, vLLM's content and reasoning, and records each model's
pair. Where vLLM's pair leaves the reasoning in the content and renderers splits
it off, a gap listed per checkpoint (``reasoning_left_in_content``), the fixture
keeps the content after vLLM's reasoning closer: there the split is renderers'
convention, not this parse's. Anywhere else the pair is wrong, and the
regeneration refuses it.
"""

import json
import os
import re
import sys
from functools import cache
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

FIXTURE = Path(__file__).parent / "fixtures" / "vllm_chat_content.json"
_REASONING_CLOSE = re.compile(r"</think(?::opensource)?>")


@cache
def _tokenizer(model):
    from renderers.base import load_tokenizer

    return load_tokenizer(model)


def replies(out_path):
    """Every reply the tests parse, with what serving it needs."""
    from parity import MODEL_CATALOG, TOOLS, kwarg_combinations
    from reference_continuations import (
        BEFORE,
        BEFORE_KWARGS,
        MESSAGES,
        SERVED_REPLIES,
        tool_reply_with_text_after,
    )

    from renderers import create_renderer
    from renderers.base import MODEL_RENDERER_MAP, vllm_parsers_for
    from renderers.configs import config_from_name

    cases = []
    for case in MODEL_CATALOG:
        name = case.resolved_renderer
        if name == "default":
            continue
        for kwargs in kwarg_combinations(case):
            for calls in [1] if name in {"llama-3", "gpt-oss"} else [1, 2]:
                reply_id = f"{case.model}-{kwargs}-calls={calls}"
                cases.append((reply_id, case, case.model, name, kwargs, calls, ""))
        if (kwargs := BEFORE_KWARGS.get(name, {})) is not None:
            reply_id = f"{case.model}-{kwargs}-{BEFORE}-calls=1"
            cases.append((reply_id, case, case.model, name, kwargs, 1, BEFORE))
    for reply_id, (model, text) in SERVED_REPLIES.items():
        cases.append((reply_id, None, model, MODEL_RENDERER_MAP[model], {}, text, ""))

    out = []
    for reply_id, case, model, name, kwargs, how, content in cases:
        tok = _tokenizer(model)
        renderer = create_renderer(
            tok, config_from_name(name), chat_template_kwargs=kwargs or None
        )
        if isinstance(how, str):
            ids = tok.encode(how, add_special_tokens=False)
        else:
            ids = tool_reply_with_text_after(
                tok, renderer, case, kwargs, MESSAGES, TOOLS, how, content
            )
        pair = vllm_parsers_for(config_from_name(name), model)
        entry = {
            "id": reply_id,
            "model": model,
            "renderer": name,
            "pair": [pair.tool, pair.reasoning],
            # A None template kwarg is unset, which the request then doesn't send.
            "chat_template_kwargs": {k: v for k, v in kwargs.items() if v is not None},
            "ids": ids,
            "stops": renderer.get_stop_token_ids(),
        }
        for label, tools in (("tools", TOOLS), ("no_tools", None)):
            prompt = renderer.render_ids(
                MESSAGES, tools=tools, add_generation_prompt=True
            )
            split = renderer.parse_response(ids, tools=tools, prompt_ids=prompt)
            entry[label] = {"prompt": prompt, "reasoning": split.reasoning_content}
        out.append(entry)
        print(reply_id, flush=True)
    Path(out_path).write_text(json.dumps(out))


def _detokenize(tok, request, prompt_ids, ids):
    """The text of each of ``ids`` as vLLM's v1 detokenizer emits it for
    ``request``: its own ``IncrementalDetokenizer``, primed with the prompt (a
    ``DecodeStream`` for a fast tokenizer, ``detokenize_incrementally`` for a
    slow one)."""
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.detokenizer import IncrementalDetokenizer

    detokenizer = IncrementalDetokenizer.from_new_request(
        tok,
        EngineCoreRequest(
            request_id="fixture",
            prompt_token_ids=list(prompt_ids),
            mm_features=None,
            sampling_params=request.to_sampling_params(len(ids) + 1, {}),
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        ),
    )
    pieces = []
    for t in ids:
        detokenizer.update([t], stop_terminated=False)
        pieces.append(detokenizer.get_next_output_text(finished=False, delta=True))
    return pieces


def _harmony_prefill(tok, prompt_ids):
    """Split renderers' gpt-oss prompt where vLLM's ends: vLLM's Harmony prompt
    stops at ``<|start|>assistant`` and the model samples the channel header,
    which renderers prefills (``<|channel|>analysis<|message|>``). vLLM parses
    the reply from that header on, so the prefill goes in front of the reply."""
    start = tok.convert_tokens_to_ids("<|start|>")
    header = max(i for i, t in enumerate(prompt_ids) if t == start) + 1
    role = tok.encode("assistant", add_special_tokens=False)
    assert prompt_ids[header : header + len(role)] == role, "no assistant header"
    split = header + len(role)
    return prompt_ids[:split], prompt_ids[split:]


def _serve(tok, model, pair, prompt_ids, ids, stop_ids, *, tools, ctk, stream):
    """(reasoning, content) of vLLM's reply to ``ids`` under tool_choice="none"."""
    from reference_continuations import MESSAGES
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser.parser_manager import ParserManager

    harmony = pair.tool == "openai"
    if harmony:
        prompt_ids, prefill = _harmony_prefill(tok, prompt_ids)
        ids = prefill + ids
    data = {"model": model, "messages": MESSAGES, "tool_choice": "none"}
    if tools:
        data["tools"] = tools
    if ctk:
        data["chat_template_kwargs"] = ctk
    request = ChatCompletionRequest.model_validate(data)
    parser_cls = ParserManager.get_parser(
        tool_parser_name=pair.tool,
        reasoning_parser_name=pair.reasoning,
        enable_auto_tools=True,
        model_name=model,
        is_harmony=harmony,
        tokenizer=tok,
    )
    template_kwargs = request.build_chat_params(None, "auto").chat_template_kwargs

    def parser():
        return parser_cls(
            tok, request.tools, chat_template_kwargs=template_kwargs, model_config=None
        )

    # online_renderer.py: a reasoning parser (or Harmony) adjusts the request.
    if harmony or parser_cls.always_adjust_request or parser_cls.reasoning_parser_cls:
        request = parser().adjust_request(request=request) or request
    stopped = bool(ids) and ids[-1] in stop_ids
    text_ids = ids[:-1] if stopped else ids
    pieces = _detokenize(tok, request, prompt_ids, text_ids)
    if not stream:
        whole = parser()
        whole.set_prompt_token_ids(list(prompt_ids))
        reasoning, content, _ = whole.parse(
            "".join(pieces), request, enable_auto_tools=True, model_output_token_ids=ids
        )
        return reasoning, content
    streaming = parser()
    steps = [(piece, [t]) for piece, t in zip(pieces, text_ids)]
    if stopped:
        steps.append(("", [ids[-1]]))
    reasoning, content = "", ""
    for n, (piece, delta_ids) in enumerate(steps):
        delta = streaming.parse_delta(
            piece, delta_ids, request, list(prompt_ids), finished=n == len(steps) - 1
        )
        if delta is not None:
            reasoning += delta.reasoning or ""
            content += delta.content or ""
    return reasoning or None, content or None


def serve(in_path):
    from parity import TOOLS
    from reference_continuations import reasoning_left_in_content, served_key

    from renderers import VLLM_CHAT_PARITY, VllmParsers

    pairs, served = {}, {}
    for reply in json.loads(Path(in_path).read_text()):
        model, ctk = reply["model"], reply["chat_template_kwargs"]
        tok = _tokenizer(model)
        pair = VllmParsers(*reply["pair"])
        pairs[model] = reply["pair"]
        for label, tools in (("tools", TOOLS), ("no_tools", None)):
            reasoning, content = _serve(
                tok,
                model,
                pair,
                reply[label]["prompt"],
                reply["ids"],
                set(reply["stops"]),
                tools=tools,
                ctk=ctk,
                stream=tools is None and pair.engine_based,
            )
            content = content or ""
            split = reply[label]["reasoning"]
            if reasoning is None and split and split.strip() in content:
                if not reasoning_left_in_content(model, ctk):
                    raise ValueError(
                        f"{reply['id']}: vLLM's pair {pair} leaves the reasoning "
                        f"in the content for {model}: fix the pair or list the "
                        "gap (reasoning_left_in_content)"
                    )
                close = _REASONING_CLOSE.search(content)
                if close is None:
                    raise ValueError(
                        f"{reply['id']}: no reasoning closer in {content!r}"
                    )
                content = content[close.end() :]
            key = served_key(model, tools, reply["ids"], ctk)
            if served.setdefault(key, (content, reasoning)) != (content, reasoning):
                raise ValueError(f"{reply['id']}: one input, two replies")
        print(reply["id"], flush=True)
    messages = sorted(set(served.values()), key=json.dumps)
    index = {message: i for i, message in enumerate(messages)}
    fixture = {
        "vllm": VLLM_CHAT_PARITY,
        "pairs": pairs,
        "messages": [{"content": c, "reasoning": r} for c, r in messages],
        "served": {key: index[message] for key, message in served.items()},
    }
    FIXTURE.write_text(json.dumps(fixture, indent=0, sort_keys=True) + "\n")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    step, path = sys.argv[1], sys.argv[2]
    {"replies": replies, "serve": serve}[step](path)
