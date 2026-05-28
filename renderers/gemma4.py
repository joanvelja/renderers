"""Gemma 4 renderer for the text chat path.

The core template is simple for ordinary chat:

    <bos><|turn>{role}\n{content}<turn|>\n

with assistant rendered as ``model``. Tool declarations/calls use a compact
Gemma-specific DSL; those paths delegate to the tokenizer template for exact
token parity and opt out of attribution masks.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    ToolSpec,
    attribute_text_segments,
    should_preserve_past_thinking,
    trim_to_turn_close,
)
from renderers.configs import Gemma4RendererConfig
from renderers.parsing import parse_gemma4

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer


class Gemma4Renderer:
    """Deterministic message → token renderer for Gemma 4 text conversations."""

    def __init__(self, tokenizer: PreTrainedTokenizer, config: Gemma4RendererConfig | None = None):
        self._tokenizer = tokenizer
        self.config = config or Gemma4RendererConfig()
        self._bos = self._token_id("<bos>")
        self._turn_start = self._token_id("<|turn>")
        self._turn_end = self._token_id("<turn|>")
        self._tool_call = self._token_id("<|tool_call>")
        self._tool_call_end = self._token_id("<tool_call|>")
        self._tool_response = self._token_id("<|tool_response>")
        self._eos = self._token_id("<eos>")

    def _token_id(self, token: str) -> int:
        tid = self._tokenizer.convert_tokens_to_ids(token)
        assert isinstance(tid, int) and tid != self._tokenizer.unk_token_id, (
            f"Special token {token!r} not found in tokenizer vocabulary"
        )
        return tid

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self._tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _uses_tooling(messages: list[Message], tools: list[ToolSpec] | None) -> bool:
        return bool(tools) or any(msg.get("tool_calls") or msg.get("role") == "tool" for msg in messages)

    @staticmethod
    def _visible_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
            return "".join(parts)
        return str(content)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        parts: list[str] = []
        for part in text.split("<channel|>"):
            if "<|channel>" in part:
                parts.append(part.split("<|channel>", 1)[0])
            else:
                parts.append(part)
        return "".join(parts).strip()

    @staticmethod
    def _last_user_index(messages: list[Message]) -> int:
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                return idx
        return -1

    @classmethod
    def _format_argument(cls, value: Any) -> str:
        if isinstance(value, str):
            return f'<|"|>{value}<|"|>'
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        if isinstance(value, int | float):
            return str(value)
        if isinstance(value, list):
            return "[" + ",".join(cls._format_argument(item) for item in value) + "]"
        if isinstance(value, dict):
            return "{" + ",".join(
                f"{key}:{cls._format_argument(item)}" for key, item in sorted(value.items())
            ) + "}"
        return cls._format_argument(str(value))

    @classmethod
    def _format_tool_response_block(cls, tool_name: str, response: Any) -> str:
        if isinstance(response, str):
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                parsed = response
        else:
            parsed = response

        if isinstance(parsed, dict):
            body = ",".join(f"{key}:{cls._format_argument(value)}" for key, value in sorted(parsed.items()))
        else:
            body = f"value:{cls._format_argument(parsed)}"
        return f"<|tool_response>response:{tool_name}{{{body}}}<tool_response|>"

    @staticmethod
    def _tool_name_for_response(source: Message, tool_msg: Message) -> str:
        if isinstance(tool_msg.get("name"), str):
            return tool_msg["name"]
        for tool_call in source.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            if tool_call.get("id") != tool_msg.get("tool_call_id"):
                continue
            function = tool_call.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                return function["name"]
        return "unknown"

    @staticmethod
    def _normalize_template_messages(messages: list[Message]) -> list[Message]:
        normalized: list[Message] = []
        for msg in messages:
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                normalized.append(msg)
                continue

            new_msg = dict(msg)
            new_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    new_calls.append(tool_call)
                    continue
                new_call = dict(tool_call)
                function = new_call.get("function")
                if isinstance(function, dict):
                    new_function = dict(function)
                    arguments = new_function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            parsed_arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            parsed_arguments = arguments
                        if isinstance(parsed_arguments, dict):
                            new_function["arguments"] = parsed_arguments
                    new_call["function"] = new_function
                new_calls.append(new_call)
            new_msg["tool_calls"] = new_calls
            normalized.append(new_msg)
        return normalized

    def _apply_template_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None,
        add_generation_prompt: bool,
    ) -> list[int]:
        kwargs: dict[str, Any] = {"add_generation_prompt": add_generation_prompt}
        if tools is not None:
            kwargs["tools"] = tools
        rendered = self._tokenizer.apply_chat_template(
            self._normalize_template_messages(messages),
            tokenize=True,
            return_dict=False,
            **kwargs,
        )
        if isinstance(rendered, dict):
            return list(rendered["input_ids"])
        return list(rendered)

    def _render_with_template(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None,
        add_generation_prompt: bool,
    ) -> RenderedTokens:
        token_ids: list[int] = []
        message_indices: list[int] = []
        prev_len = 0

        for idx in range(len(messages)):
            cur_ids = self._apply_template_ids(
                messages[: idx + 1], tools=tools, add_generation_prompt=False
            )
            message_indices.extend([idx] * (len(cur_ids) - prev_len))
            token_ids = cur_ids
            prev_len = len(cur_ids)

        if add_generation_prompt:
            token_ids = self._apply_template_ids(
                messages, tools=tools, add_generation_prompt=True
            )
            message_indices.extend([-1] * (len(token_ids) - prev_len))

        return RenderedTokens(
            token_ids=token_ids,
            message_indices=message_indices,
            message_roles=[m.get("role") or "" for m in messages],
        )

    def _render_text_chat(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool,
        include_bos: bool,
    ) -> RenderedTokens:
        tokens: list[int] = []
        indices: list[int] = []
        sampled: list[bool] = []
        content_mask: list[bool] = []

        def emit_special(token_id: int, msg_idx: int, *, is_sampled: bool, is_content: bool) -> None:
            tokens.append(token_id)
            indices.append(msg_idx)
            sampled.append(is_sampled)
            content_mask.append(is_content)

        def emit_text(text: str, msg_idx: int, *, is_sampled: bool, is_content: bool) -> None:
            ids = self._encode(text)
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))
            sampled.extend([is_sampled] * len(ids))
            content_mask.extend([is_content] * len(ids))

        def emit_text_segments(segments: list[tuple[str, bool]], msg_idx: int, *, is_sampled: bool) -> None:
            for token_id, is_content in attribute_text_segments(self._tokenizer, segments):
                tokens.append(token_id)
                indices.append(msg_idx)
                sampled.append(is_sampled and is_content)
                content_mask.append(is_content)

        if include_bos:
            emit_special(self._bos, -1, is_sampled=False, is_content=False)

        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role not in {"system", "user", "assistant"}:
                continue

            rendered_role = "model" if role == "assistant" else role
            content = msg.get("content") if isinstance(msg.get("content"), str) else ""
            is_assistant = role == "assistant"
            preserve_thinking = is_assistant and should_preserve_past_thinking(
                messages,
                idx,
                preserve_all_thinking=self.config.preserve_all_thinking,
                preserve_thinking_between_tool_calls=self.config.preserve_thinking_between_tool_calls,
            )
            reasoning = msg.get("reasoning") or msg.get("reasoning_content")

            emit_special(self._turn_start, idx, is_sampled=False, is_content=False)
            segments = [(rendered_role + "\n", False)]
            if preserve_thinking and isinstance(reasoning, str) and reasoning:
                segments.extend(
                    [
                        ("<|channel>thought\n", False),
                        (reasoning, True),
                        ("\n<channel|>", False),
                    ]
                )
            if content:
                segments.append((content, True))
            emit_text_segments(segments, idx, is_sampled=is_assistant)
            emit_special(self._turn_end, idx, is_sampled=is_assistant, is_content=is_assistant)
            emit_text("\n", idx, is_sampled=False, is_content=False)

        if add_generation_prompt:
            emit_special(self._turn_start, -1, is_sampled=False, is_content=False)
            emit_text("model\n", -1, is_sampled=False, is_content=False)

        return RenderedTokens(
            token_ids=tokens,
            message_indices=indices,
            sampled_mask=sampled,
            is_content=content_mask,
            message_roles=[m.get("role") or "" for m in messages],
        )

    def _render_tool_chat_preserving_thinking(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None,
        add_generation_prompt: bool,
    ) -> RenderedTokens:
        tokens: list[int] = []
        indices: list[int] = []
        sampled: list[bool] = []
        content_mask: list[bool] = []

        def emit_text(text: str, msg_idx: int, *, is_sampled: bool, is_content: bool) -> None:
            ids = self._encode(text)
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))
            sampled.extend([is_sampled] * len(ids))
            content_mask.extend([is_content] * len(ids))

        def emit_segments(segments: list[tuple[str, bool]], msg_idx: int, *, is_sampled: bool) -> None:
            for token_id, is_content in attribute_text_segments(self._tokenizer, segments):
                tokens.append(token_id)
                indices.append(msg_idx)
                sampled.append(is_sampled and is_content)
                content_mask.append(is_content)

        emit_text("<bos>", -1, is_sampled=False, is_content=False)

        start_idx = 0
        first_role = messages[0].get("role") if messages else None
        if tools or first_role in {"system", "developer"}:
            emit_text("<|turn>system\n", -1, is_sampled=False, is_content=False)
            if messages and first_role in {"system", "developer"}:
                emit_text(self._visible_text(messages[0].get("content")).strip(), 0, is_sampled=False, is_content=True)
                start_idx = 1
            if tools:
                for tool in tools:
                    emit_text("<|tool>", -1, is_sampled=False, is_content=False)
                    emit_text(self._format_argument(tool), -1, is_sampled=False, is_content=False)
                    emit_text("<tool|>", -1, is_sampled=False, is_content=False)
            emit_text("<turn|>\n", -1, is_sampled=False, is_content=False)

        last_user = self._last_user_index(messages)
        idx = start_idx
        while idx < len(messages):
            msg = messages[idx]
            role = msg.get("role")
            if role == "tool":
                idx += 1
                continue

            rendered_role = "model" if role == "assistant" else role
            is_assistant = role == "assistant"
            emit_text(f"<|turn>{rendered_role}\n", idx, is_sampled=False, is_content=False)

            if is_assistant:
                reasoning = msg.get("reasoning") or msg.get("reasoning_content")
                preserve_thinking = should_preserve_past_thinking(
                    messages,
                    idx,
                    preserve_all_thinking=self.config.preserve_all_thinking,
                    preserve_thinking_between_tool_calls=self.config.preserve_thinking_between_tool_calls,
                )
                native_thinking = idx > last_user and bool(msg.get("tool_calls"))
                if (native_thinking or preserve_thinking) and isinstance(reasoning, str) and reasoning:
                    emit_segments(
                        [
                            ("<|channel>thought\n", False),
                            (reasoning, True),
                            ("\n<channel|>", False),
                        ],
                        idx,
                        is_sampled=True,
                    )

                for tool_call in msg.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function")
                    if not isinstance(function, dict):
                        continue
                    name = function.get("name")
                    if not isinstance(name, str):
                        continue
                    arguments = function.get("arguments") or {}
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            pass
                    emit_text(
                        f"<|tool_call>call:{name}{self._format_argument(arguments)}<tool_call|>",
                        idx,
                        is_sampled=True,
                        is_content=True,
                    )

                scan_idx = idx + 1
                saw_tool_response = False
                while scan_idx < len(messages) and messages[scan_idx].get("role") == "tool":
                    tool_msg = messages[scan_idx]
                    emit_text(
                        self._format_tool_response_block(
                            self._tool_name_for_response(msg, tool_msg),
                            tool_msg.get("content"),
                        ),
                        scan_idx,
                        is_sampled=False,
                        is_content=True,
                    )
                    saw_tool_response = True
                    scan_idx += 1

                content = self._strip_thinking(self._visible_text(msg.get("content")))
                if content:
                    emit_text(content, idx, is_sampled=True, is_content=True)
                if msg.get("tool_calls") and not saw_tool_response:
                    emit_text("<|tool_response>", idx, is_sampled=True, is_content=False)
                else:
                    emit_text("<turn|>\n", idx, is_sampled=is_assistant, is_content=is_assistant)
                idx = scan_idx
                continue

            content = self._visible_text(msg.get("content")).strip()
            if content:
                emit_text(content, idx, is_sampled=False, is_content=True)
            emit_text("<turn|>\n", idx, is_sampled=False, is_content=False)
            idx += 1

        if add_generation_prompt:
            emit_text("<|turn>model\n", -1, is_sampled=False, is_content=False)

        return RenderedTokens(
            token_ids=tokens,
            message_indices=indices,
            sampled_mask=sampled,
            is_content=content_mask,
            message_roles=[m.get("role") or "" for m in messages],
        )

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if self._uses_tooling(messages, tools):
            if self.config.preserve_all_thinking or self.config.preserve_thinking_between_tool_calls:
                return self._render_tool_chat_preserving_thinking(
                    messages,
                    tools=tools,
                    add_generation_prompt=add_generation_prompt,
                )
            return self._render_with_template(messages, tools=tools, add_generation_prompt=add_generation_prompt)
        return self._render_text_chat(messages, add_generation_prompt=add_generation_prompt, include_bos=True)

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return self.render(messages, tools=tools, add_generation_prompt=add_generation_prompt).token_ids

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,  # noqa: ARG002 - Gemma4 call syntax is self-describing.
    ) -> ParsedResponse:
        return parse_gemma4(
            self._tokenizer,
            token_ids,
            stop_ids={self._turn_end, self._eos, self._tool_response},
            tool_call_id=self._tool_call,
            tool_call_end_id=self._tool_call_end,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._turn_end, self._eos, self._tool_response]

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> RenderedTokens | None:
        if not previous_prompt_ids or not new_messages or self._uses_tooling(new_messages, tools):
            return None
        if any(msg.get("role") == "assistant" for msg in new_messages):
            return None

        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._turn_end, self._eos},
            synthesize_close=self._turn_end,
        )
        if previous_ids is None:
            return None

        bridge = self._render_text_chat(new_messages, add_generation_prompt=True, include_bos=False)
        bridge_prefix = self._encode("\n")
        extension_ids = bridge_prefix + bridge.token_ids
        total_len = len(previous_ids) + len(extension_ids)
        return RenderedTokens(
            token_ids=previous_ids + extension_ids,
            message_indices=[-1] * (len(previous_ids) + len(bridge_prefix)) + bridge.message_indices,
            sampled_mask=[False] * total_len,
            is_content=[False] * (len(previous_ids) + len(bridge_prefix)) + bridge.is_content,
            message_roles=bridge.message_roles,
        )
