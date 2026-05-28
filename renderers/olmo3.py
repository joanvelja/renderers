"""OLMo 3 Renderer — hard-coded Python mirroring the OLMo 3 chat template."""

from __future__ import annotations

import json
from typing import Any

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    ToolSpec,
    attribute_text_segments,
    reject_assistant_in_extension,
    trim_to_turn_close,
)
from renderers.configs import Olmo3RendererConfig
from renderers.parsing import parse_olmo3

_DEFAULT_SYSTEM_NO_TOOLS = (
    "You are a helpful function-calling AI assistant. "
    "You do not currently have access to any functions. <functions></functions>"
)
_DEFAULT_SYSTEM_TOOLS_PREFIX = (
    "You are a helpful function-calling AI assistant. "
    "You are provided with function signatures within <functions></functions> "
    "XML tags. You may call one or more functions to assist with the user query. "
    "Output any function calls within <function_calls></function_calls> XML tags. "
    "Do not make assumptions about what values to plug into functions."
)


class Olmo3Renderer:
    """Deterministic message → token renderer for OLMo 3 models."""

    def __init__(self, tokenizer, config: Olmo3RendererConfig | None = None):
        self._tokenizer = tokenizer
        self.config = config or Olmo3RendererConfig()

        self._im_start = self._token_id("<|im_start|>")
        self._im_end = self._token_id("<|im_end|>")
        self._endoftext = self._token_id("<|endoftext|>")
        self._function_calls = self._token_id("<function_calls>")
        self._function_calls_end = self._token_id("</function_calls>")

    def _token_id(self, token: str) -> int:
        vocab = self._tokenizer.get_vocab()
        if token in vocab:
            return int(vocab[token])
        tid = self._tokenizer.convert_tokens_to_ids(token)
        assert isinstance(tid, int) and tid != self._tokenizer.unk_token_id, (
            f"Special token {token!r} not found in tokenizer vocabulary"
        )
        return tid

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self._tokenizer.encode(text, add_special_tokens=False)

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        tokens: list[int] = []
        indices: list[int] = []
        sampled: list[bool] = []
        content_mask: list[bool] = []

        def emit_special(
            token_id: int, msg_idx: int, *, is_sampled: bool, is_content: bool
        ) -> None:
            tokens.append(token_id)
            indices.append(msg_idx)
            sampled.append(is_sampled)
            content_mask.append(is_content)

        def emit_text(
            text: str, msg_idx: int, *, is_sampled: bool, is_content: bool
        ) -> None:
            ids = self._encode(text)
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))
            sampled.extend([is_sampled] * len(ids))
            content_mask.extend([is_content] * len(ids))

        def emit_text_segments(
            segments: list[tuple[str, bool]], msg_idx: int, *, is_sampled: bool
        ) -> None:
            for token_id, is_content in attribute_text_segments(
                self._tokenizer, segments
            ):
                tokens.append(token_id)
                indices.append(msg_idx)
                sampled.append(is_sampled)
                content_mask.append(is_content)

        first_is_system = bool(messages) and messages[0].get("role") == "system"
        if not first_is_system:
            self._render_default_system(
                tools,
                emit_special=emit_special,
                emit_text=emit_text,
            )

        num_messages = len(messages)
        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content") if isinstance(msg.get("content"), str) else ""
            if role == "system":
                self._render_system(
                    i,
                    content,
                    tools,
                    functions=msg.get("functions"),
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_text_segments=emit_text_segments,
                )
            elif role == "user":
                self._render_plain_turn(
                    "user",
                    i,
                    content,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_text_segments=emit_text_segments,
                )
            elif role == "assistant":
                self._render_assistant(
                    msg,
                    i,
                    content,
                    is_last=i == num_messages - 1,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )
            elif role in {"tool", "environment"}:
                self._render_plain_turn(
                    "environment",
                    i,
                    content,
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_text_segments=emit_text_segments,
                )

            if i == num_messages - 1 and add_generation_prompt:
                emit_special(self._im_start, -1, is_sampled=False, is_content=False)
                emit_text("assistant\n", -1, is_sampled=False, is_content=False)

        return RenderedTokens(
            token_ids=tokens,
            message_indices=indices,
            sampled_mask=sampled,
            is_content=content_mask,
            message_roles=[m.get("role") or "" for m in messages],
        )

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return self.render(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        ).token_ids

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,  # noqa: ARG002 - OLMo3 calls carry their own arg literals.
    ) -> ParsedResponse:
        return parse_olmo3(
            self._tokenizer,
            token_ids,
            stop_ids={self._im_end, self._endoftext},
            function_calls_id=self._function_calls,
            function_calls_end_id=self._function_calls_end,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._im_end, self._endoftext]

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> RenderedTokens | None:
        if (
            not previous_prompt_ids
            or not new_messages
            or reject_assistant_in_extension(new_messages)
        ):
            return None

        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._im_end, self._endoftext},
            synthesize_close=self._im_end,
        )
        if previous_ids is None:
            return None

        bridge = self.render(new_messages, tools=tools, add_generation_prompt=True)
        bridge_prefix = self._encode("\n")
        extension_ids = bridge_prefix + bridge.token_ids
        total_len = len(previous_ids) + len(extension_ids)
        return RenderedTokens(
            token_ids=previous_ids + extension_ids,
            message_indices=[-1] * (len(previous_ids) + len(bridge_prefix))
            + bridge.message_indices,
            sampled_mask=[False] * total_len,
            is_content=[False] * (len(previous_ids) + len(bridge_prefix))
            + bridge.is_content,
            message_roles=bridge.message_roles,
        )

    def _render_default_system(self, tools, *, emit_special, emit_text) -> None:
        emit_special(self._im_start, -1, is_sampled=False, is_content=False)
        emit_text("system\n", -1, is_sampled=False, is_content=False)
        if tools:
            emit_text(
                _DEFAULT_SYSTEM_TOOLS_PREFIX,
                -1,
                is_sampled=False,
                is_content=False,
            )
            emit_text(
                "<functions>" + _tojson(tools) + "</functions>",
                -1,
                is_sampled=False,
                is_content=False,
            )
        else:
            emit_text(
                _DEFAULT_SYSTEM_NO_TOOLS,
                -1,
                is_sampled=False,
                is_content=False,
            )
        emit_special(self._im_end, -1, is_sampled=False, is_content=False)
        emit_text("\n", -1, is_sampled=False, is_content=False)

    def _render_system(
        self,
        msg_idx: int,
        content: str,
        tools,
        *,
        functions,
        emit_special,
        emit_text,
        emit_text_segments,
    ) -> None:
        emit_special(self._im_start, msg_idx, is_sampled=False, is_content=False)
        segments = [("system\n", False)]
        if content:
            segments.append((content, True))
        emit_text_segments(segments, msg_idx, is_sampled=False)
        if tools is not None:
            emit_text(
                "<functions>" + _tojson(tools) + "</functions>",
                msg_idx,
                is_sampled=False,
                is_content=False,
            )
        elif functions is not None:
            emit_text(
                " <functions>" + str(functions) + "</functions>",
                msg_idx,
                is_sampled=False,
                is_content=False,
            )
        emit_special(self._im_end, msg_idx, is_sampled=False, is_content=False)
        emit_text("\n", msg_idx, is_sampled=False, is_content=False)

    def _render_plain_turn(
        self,
        role: str,
        msg_idx: int,
        content: str,
        *,
        emit_special,
        emit_text,
        emit_text_segments,
    ) -> None:
        emit_special(self._im_start, msg_idx, is_sampled=False, is_content=False)
        segments = [(role + "\n", False)]
        if content:
            segments.append((content, True))
        emit_text_segments(segments, msg_idx, is_sampled=False)
        emit_special(self._im_end, msg_idx, is_sampled=False, is_content=False)
        emit_text("\n", msg_idx, is_sampled=False, is_content=False)

    def _render_assistant(
        self,
        msg: Message,
        msg_idx: int,
        content: str,
        *,
        is_last: bool,
        emit_special,
        emit_text,
    ) -> None:
        emit_special(self._im_start, msg_idx, is_sampled=False, is_content=False)
        emit_text("assistant\n", msg_idx, is_sampled=False, is_content=False)
        if content:
            emit_text(content, msg_idx, is_sampled=True, is_content=True)
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            emit_special(
                self._function_calls, msg_idx, is_sampled=True, is_content=True
            )
            for tc_idx, tool_call in enumerate(tool_calls):
                emit_text(
                    _format_tool_call(tool_call),
                    msg_idx,
                    is_sampled=True,
                    is_content=True,
                )
                if tc_idx != len(tool_calls) - 1:
                    emit_text("\n", msg_idx, is_sampled=True, is_content=True)
            emit_special(
                self._function_calls_end,
                msg_idx,
                is_sampled=True,
                is_content=True,
            )
        if is_last:
            emit_special(self._endoftext, msg_idx, is_sampled=True, is_content=True)
        else:
            emit_special(self._im_end, msg_idx, is_sampled=True, is_content=True)
            emit_text("\n", msg_idx, is_sampled=False, is_content=False)


def _tojson(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _format_tool_call(tool_call: Any) -> str:
    if not isinstance(tool_call, dict):
        return str(tool_call)
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else tool_call
    name = function.get("name", "")
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = {}
        arguments = parsed if isinstance(parsed, dict) else {}
    parts = [f"{key}={_tojson(value)}" for key, value in arguments.items()]
    return f"{name}({', '.join(parts)})"
