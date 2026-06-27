"""Gemma 4 renderer for text and tool-chat conversations.

Gemma 4 is a multimodal family, but this renderer intentionally implements
only the text/tool chat-template path. Image/audio/video content parts are
rejected until a multimodal sidecar is implemented, so auto-routing never
silently drops non-text inputs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from renderers.base import (
    Message,
    ParsedResponse,
    RenderedTokens,
    ToolSpec,
    _get_offset_tokenizer,
    extract_message_tool_names,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
    trim_to_turn_close,
)
from renderers.configs import Gemma4RendererConfig
from renderers.parsing import parse_gemma4

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer


_QUOTE = '<|"|>'
_MULTIMODAL_PART_TYPES = {
    "image",
    "image_url",
    "audio",
    "audio_url",
    "video",
    "video_url",
}
_EMPTY_THOUGHT_GENERATION_PROMPT_MODELS = frozenset(
    {
        "google/gemma-4-31B",
        "google/gemma-4-31B-it",
        "google/gemma-4-26B-A4B",
        "google/gemma-4-26B-A4B-it",
    }
)


@dataclass(frozen=True)
class _Segment:
    text: str
    msg_idx: int
    is_sampled: bool
    is_content: bool


class Gemma4Renderer:
    """Deterministic message -> token renderer for the Gemma 4 chat template."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: Gemma4RendererConfig | None = None,
    ):
        self._tokenizer = tokenizer
        self.config = config or Gemma4RendererConfig()
        # Default to retaining thinking across turns: the bridge reuses the prior
        # prefix verbatim (token-faithful prefix continuation / stable KV cache).
        # Configurable via thinking_retention (e.g. "tool_cycle" to drop at a new
        # user-query boundary).
        self.effective_thinking_retention = resolve_thinking_retention(
            self.config, "all"
        )
        self._turn_end = self._token_id("<turn|>")
        self._tool_call = self._token_id("<|tool_call>")
        self._tool_call_end = self._token_id("<tool_call|>")
        self._tool_response = self._token_id("<|tool_response>")
        self._eos = self._token_id("<eos>")
        self._add_empty_thought_generation_prompt = self._detect_empty_thought_prompt()

    def _detect_empty_thought_prompt(self) -> bool:
        template = getattr(self._tokenizer, "chat_template", "") or ""
        if "<|channel>thought\\n<channel|>" in template:
            return True

        # Base Gemma4 checkpoints do not currently ship chat templates, but
        # their size-paired IT checkpoints do. Match that generation-prompt
        # split when routing base checkpoints through this renderer. Keep this
        # exact-match, mirroring MODEL_RENDERER_MAP / Qwen3.5 size defaults.
        model_name = str(getattr(self._tokenizer, "name_or_path", ""))
        return model_name in _EMPTY_THOUGHT_GENERATION_PROMPT_MODELS

    def _token_id(self, token: str) -> int:
        tid = self._tokenizer.convert_tokens_to_ids(token)
        assert isinstance(tid, int) and tid != self._tokenizer.unk_token_id, (
            f"Special token {token!r} not found in tokenizer vocabulary"
        )
        return tid

    @staticmethod
    def _uses_tooling(messages: list[Message], tools: list[ToolSpec] | None) -> bool:
        return bool(tools) or any(
            msg.get("tool_calls")
            or msg.get("tool_responses")
            or msg.get("role") == "tool"
            for msg in messages
        )

    @staticmethod
    def _reject_multimodal(part_type: str) -> None:
        if part_type in _MULTIMODAL_PART_TYPES:
            raise ValueError(
                "Gemma4Renderer currently supports only text/tool chat; "
                f"{part_type!r} content parts require multimodal sidecar support."
            )

    @classmethod
    def _content_text(cls, content: Any, *, role: str) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            text = cls._strip_thinking(content) if role == "assistant" else content
            return text.strip()
        if isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text = item
                elif isinstance(item, Mapping):
                    part_type = item.get("type")
                    if isinstance(part_type, str):
                        cls._reject_multimodal(part_type)
                    if part_type != "text":
                        continue
                    text = str(item.get("text") or "")
                else:
                    continue
                if role == "assistant":
                    parts.append(cls._strip_thinking(text))
                else:
                    parts.append(text.strip())
            return "".join(parts)
        return str(content).strip()

    @classmethod
    def _system_text(cls, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            parts: list[str] = []
            for item in content:
                if isinstance(item, Mapping):
                    part_type = item.get("type")
                    if isinstance(part_type, str):
                        cls._reject_multimodal(part_type)
                    if "text" in item:
                        parts.append(str(item.get("text") or "").strip() + " ")
                elif isinstance(item, str):
                    parts.append(item.strip() + " ")
            return "".join(parts)
        return str(content).strip()

    @classmethod
    def _tool_text(cls, content: Any) -> Any:
        if isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, Mapping):
                    part_type = item.get("type")
                    if isinstance(part_type, str):
                        cls._reject_multimodal(part_type)
                    if part_type == "text":
                        text_parts.append(str(item.get("text") or ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            return "".join(text_parts)
        return content

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
    def _format_argument(cls, value: Any, *, escape_keys: bool = True) -> str:
        if isinstance(value, str):
            return f"{_QUOTE}{value}{_QUOTE}"
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "None"
        if isinstance(value, int | float):
            return str(value)
        if isinstance(value, Mapping):
            items = []
            for key, item in sorted(value.items()):
                key_text = f"{_QUOTE}{key}{_QUOTE}" if escape_keys else str(key)
                items.append(
                    f"{key_text}:{cls._format_argument(item, escape_keys=escape_keys)}"
                )
            return "{" + ",".join(items) + "}"
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return (
                "["
                + ",".join(
                    cls._format_argument(item, escape_keys=escape_keys)
                    for item in value
                )
                + "]"
            )
        return str(value)

    @classmethod
    def _format_parameters(
        cls,
        properties: Mapping[str, Any],
        required: Sequence[str] | None,
        *,
        filter_keys: bool = False,
    ) -> str:
        standard_keys = {"description", "type", "properties", "required", "nullable"}
        out: list[str] = []
        for key, raw_value in sorted(properties.items()):
            if filter_keys and key in standard_keys:
                continue
            if not isinstance(raw_value, Mapping):
                continue
            value = raw_value
            pieces: list[str] = []
            description = value.get("description")
            if description:
                pieces.append(f"description:{_QUOTE}{description}{_QUOTE}")

            type_value = value.get("type")
            type_upper = str(type_value).upper() if type_value is not None else ""
            if type_upper == "STRING" and value.get("enum"):
                pieces.append(f"enum:{cls._format_argument(value['enum'])}")
            elif type_upper == "ARRAY":
                items = value.get("items")
                if isinstance(items, Mapping) and items:
                    item_pieces: list[str] = []
                    for item_key, item_value in sorted(items.items()):
                        if item_value is None:
                            continue
                        if item_key == "properties" and isinstance(item_value, Mapping):
                            nested_required = (
                                items.get("required")
                                if isinstance(items.get("required"), Sequence)
                                else []
                            )
                            item_pieces.append(
                                "properties:{"
                                + cls._format_parameters(item_value, nested_required)
                                + "}"
                            )
                        elif item_key == "required" and isinstance(
                            item_value, Sequence
                        ):
                            item_pieces.append(
                                "required:["
                                + ",".join(
                                    f"{_QUOTE}{req}{_QUOTE}" for req in item_value
                                )
                                + "]"
                            )
                        elif item_key == "type":
                            if isinstance(item_value, str):
                                rendered_type = cls._format_argument(item_value.upper())
                            elif isinstance(item_value, Sequence):
                                rendered_type = cls._format_argument(
                                    [str(v).upper() for v in item_value]
                                )
                            else:
                                rendered_type = cls._format_argument(item_value)
                            item_pieces.append(f"type:{rendered_type}")
                        else:
                            item_pieces.append(
                                f"{item_key}:{cls._format_argument(item_value)}"
                            )
                    pieces.append("items:{" + ",".join(item_pieces) + "}")

            if value.get("nullable"):
                pieces.append("nullable:true")

            if type_upper == "OBJECT":
                nested_props = value.get("properties")
                if isinstance(nested_props, Mapping):
                    nested_required = (
                        value.get("required")
                        if isinstance(value.get("required"), Sequence)
                        else []
                    )
                    pieces.append(
                        "properties:{"
                        + cls._format_parameters(nested_props, nested_required)
                        + "}"
                    )
                elif isinstance(value, Mapping):
                    nested_required = (
                        value.get("required")
                        if isinstance(value.get("required"), Sequence)
                        else []
                    )
                    pieces.append(
                        "properties:{"
                        + cls._format_parameters(
                            value,
                            nested_required,
                            filter_keys=True,
                        )
                        + "}"
                    )
                if value.get("required"):
                    pieces.append(
                        "required:["
                        + ",".join(
                            f"{_QUOTE}{item}{_QUOTE}"
                            for item in value.get("required", [])
                        )
                        + "]"
                    )

            pieces.append(f"type:{_QUOTE}{type_upper}{_QUOTE}")
            out.append(f"{key}:{{{','.join(pieces)}}}")
        return ",".join(out)

    @classmethod
    def _format_function_declaration(cls, tool_data: Mapping[str, Any]) -> str:
        function = tool_data.get("function", tool_data)
        if not isinstance(function, Mapping):
            raise ValueError(
                f"Gemma4 tool declaration must be a mapping: {tool_data!r}"
            )
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"Gemma4 tool declaration missing function name: {tool_data!r}"
            )
        description = function.get("description") or ""
        out = [f"declaration:{name}{{description:{_QUOTE}{description}{_QUOTE}"]

        params = function.get("parameters")
        if isinstance(params, Mapping) and params:
            param_pieces: list[str] = []
            props = params.get("properties")
            required = (
                params.get("required")
                if isinstance(params.get("required"), Sequence)
                else []
            )
            if isinstance(props, Mapping) and props:
                param_pieces.append(
                    "properties:{" + cls._format_parameters(props, required) + "}"
                )
            if required:
                param_pieces.append(
                    "required:["
                    + ",".join(f"{_QUOTE}{item}{_QUOTE}" for item in required)
                    + "]"
                )
            param_type = params.get("type")
            if param_type:
                param_pieces.append(f"type:{_QUOTE}{str(param_type).upper()}{_QUOTE}")
            out.append(",parameters:{" + ",".join(param_pieces) + "}")

        response = function.get("response")
        response_is_object = False
        if isinstance(response, Mapping):
            response_text = ""
            if response.get("description"):
                response_text += (
                    f"description:{_QUOTE}{response['description']}{_QUOTE},"
                )
            if str(response.get("type", "")).upper() == "OBJECT":
                response_is_object = True
                response_text += f"type:{_QUOTE}{str(response['type']).upper()}{_QUOTE}"
            out.append(",response:{" + response_text + "}")

        if not isinstance(response, Mapping) or response_is_object:
            out.append("}")
        return "".join(out)

    @classmethod
    def _format_tool_response_segments(
        cls,
        tool_name: str,
        response: Any,
    ) -> list[tuple[str, bool]]:
        response = cls._tool_text(response)
        if isinstance(response, Mapping):
            body = ",".join(
                f"{key}:{cls._format_argument(value, escape_keys=False)}"
                for key, value in sorted(response.items())
            )
        else:
            body = f"value:{cls._format_argument(response, escape_keys=False)}"
        return [
            (f"<|tool_response>response:{tool_name}{{", False),
            (body, True),
            ("}<tool_response|>", False),
        ]

    @staticmethod
    def _tool_name_for_response(source: Message, tool_msg: Message) -> str:
        name = tool_msg.get("name")
        resolved = name if isinstance(name, str) and name else "unknown"
        for tool_call in source.get("tool_calls") or []:
            if not isinstance(tool_call, Mapping):
                continue
            if tool_call.get("id") != tool_msg.get("tool_call_id"):
                continue
            function = tool_call.get("function")
            if isinstance(function, Mapping) and isinstance(function.get("name"), str):
                resolved = function["name"]
        return resolved

    @staticmethod
    def _normalize_arguments(arguments: Any) -> Any:
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return arguments
            if isinstance(parsed, Mapping):
                return parsed
        return arguments

    def _segments_to_rendered(
        self,
        segments: list[_Segment],
        *,
        message_roles: list[str],
        message_tool_names: list[str | None],
    ) -> RenderedTokens:
        text = "".join(segment.text for segment in segments)
        if not text:
            return RenderedTokens(
                message_roles=message_roles,
                message_tool_names=message_tool_names,
            )

        offset_tokenizer = _get_offset_tokenizer(self._tokenizer)
        encoding = offset_tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = list(encoding["input_ids"])
        offsets = list(encoding["offset_mapping"])

        spans: list[tuple[int, int, _Segment]] = []
        pos = 0
        for segment in segments:
            end = pos + len(segment.text)
            if end > pos:
                spans.append((pos, end, segment))
            pos = end
        if not spans:
            return RenderedTokens(
                message_roles=message_roles,
                message_tool_names=message_tool_names,
            )

        token_segments: list[_Segment] = []
        span_idx = 0
        last_segment = spans[-1][2]
        total_len = pos
        for start, _end in offsets:
            if start >= total_len:
                token_segments.append(last_segment)
                continue
            while span_idx < len(spans) and start >= spans[span_idx][1]:
                span_idx += 1
            if (
                span_idx < len(spans)
                and spans[span_idx][0] <= start < spans[span_idx][1]
            ):
                token_segments.append(spans[span_idx][2])
            else:
                token_segments.append(last_segment)

        return RenderedTokens(
            token_ids=token_ids,
            message_indices=[segment.msg_idx for segment in token_segments],
            sampled_mask=[segment.is_sampled for segment in token_segments],
            is_content=[segment.is_content for segment in token_segments],
            message_roles=message_roles,
            message_tool_names=message_tool_names,
        )

    def _build_segments(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None,
        add_generation_prompt: bool,
        include_bos: bool,
        include_initial_block: bool,
    ) -> list[_Segment]:
        segments: list[_Segment] = []

        def emit(
            text: str, msg_idx: int, *, is_sampled: bool, is_content: bool
        ) -> None:
            if text:
                segments.append(_Segment(text, msg_idx, is_sampled, is_content))

        if include_bos:
            emit("<bos>", -1, is_sampled=False, is_content=False)

        start_idx = 0
        first_role = messages[0].get("role") if messages else None
        if include_initial_block and (
            self.config.enable_thinking
            or bool(tools)
            or first_role in {"system", "developer"}
        ):
            emit("<|turn>system\n", -1, is_sampled=False, is_content=False)
            if self.config.enable_thinking:
                emit("<|think|>\n", -1, is_sampled=False, is_content=False)
            if messages and first_role in {"system", "developer"}:
                text = self._system_text(messages[0].get("content"))
                emit(text, 0, is_sampled=False, is_content=bool(text))
                start_idx = 1
            if tools:
                for tool in tools:
                    declaration = self._format_function_declaration(tool)
                    emit(
                        f"<|tool>{declaration}<tool|>",
                        -1,
                        is_sampled=False,
                        is_content=False,
                    )
            emit("<turn|>\n", -1, is_sampled=False, is_content=False)

        loop_messages = messages[start_idx:]
        last_user = self._last_user_index(loop_messages)
        prev_message_type: str | None = None

        for local_idx, msg in enumerate(loop_messages):
            msg_idx = start_idx + local_idx
            role = msg.get("role")
            if role == "tool":
                continue

            prev_message_type = None
            rendered_role = "model" if role == "assistant" else str(role)

            prev_non_tool_role: str | None = None
            for prev in reversed(loop_messages[:local_idx]):
                if prev.get("role") != "tool":
                    prev_non_tool_role = prev.get("role")
                    break
            continue_same_model_turn = (
                rendered_role == "model" and prev_non_tool_role == "assistant"
            )
            if not continue_same_model_turn:
                emit(
                    f"<|turn>{rendered_role}\n",
                    msg_idx,
                    is_sampled=False,
                    is_content=False,
                )

            is_assistant = role == "assistant"
            tool_calls = msg.get("tool_calls") or []
            thinking_text = msg.get("reasoning") or msg.get("reasoning_content")
            # render() is retention-agnostic (canonical thinking_retention model,
            # matching upstream renderers): only the current tool cycle's reasoning
            # is emitted inline; cross-turn retention is handled by
            # bridge_to_next_turn via should_rerender_for_thinking_retention.
            native_thinking = (
                is_assistant
                and isinstance(thinking_text, str)
                and local_idx > last_user
                and bool(tool_calls)
            )
            if native_thinking and thinking_text:
                emit("<|channel>thought\n", msg_idx, is_sampled=True, is_content=True)
                emit(thinking_text, msg_idx, is_sampled=True, is_content=True)
                emit("\n<channel|>", msg_idx, is_sampled=True, is_content=True)

            if is_assistant and tool_calls:
                for tool_call in tool_calls:
                    if not isinstance(tool_call, Mapping):
                        continue
                    function = tool_call.get("function")
                    if not isinstance(function, Mapping):
                        continue
                    name = function.get("name")
                    if not isinstance(name, str) or not name:
                        continue
                    arguments = self._normalize_arguments(
                        function.get("arguments") or {}
                    )
                    if isinstance(arguments, Mapping):
                        args_text = ",".join(
                            f"{key}:{self._format_argument(value, escape_keys=False)}"
                            for key, value in sorted(arguments.items())
                        )
                    elif isinstance(arguments, str):
                        args_text = arguments
                    else:
                        args_text = str(arguments)
                    emit(
                        f"<|tool_call>call:{name}{{{args_text}}}<tool_call|>",
                        msg_idx,
                        is_sampled=True,
                        is_content=True,
                    )
                prev_message_type = "tool_call"

            saw_tool_response = False
            if is_assistant and msg.get("tool_responses"):
                for tool_response in msg.get("tool_responses") or []:
                    if not isinstance(tool_response, Mapping):
                        continue
                    name = tool_response.get("name")
                    tool_name = name if isinstance(name, str) and name else "unknown"
                    for text, is_content in self._format_tool_response_segments(
                        tool_name,
                        tool_response.get("response"),
                    ):
                        emit(text, msg_idx, is_sampled=False, is_content=is_content)
                    saw_tool_response = True
                    prev_message_type = "tool_response"
            elif is_assistant and tool_calls:
                scan_idx = local_idx + 1
                while scan_idx < len(loop_messages):
                    follow = loop_messages[scan_idx]
                    if follow.get("role") != "tool":
                        break
                    tool_msg_idx = start_idx + scan_idx
                    tool_name = self._tool_name_for_response(msg, follow)
                    for text, is_content in self._format_tool_response_segments(
                        tool_name,
                        follow.get("content"),
                    ):
                        emit(
                            text, tool_msg_idx, is_sampled=False, is_content=is_content
                        )
                    saw_tool_response = True
                    prev_message_type = "tool_response"
                    scan_idx += 1

            content = self._content_text(msg.get("content"), role=str(role))
            if content:
                emit(content, msg_idx, is_sampled=is_assistant, is_content=True)

            has_content = bool(content.strip())
            if prev_message_type == "tool_call" and not saw_tool_response:
                emit("<|tool_response>", msg_idx, is_sampled=True, is_content=False)
            elif not (saw_tool_response and not has_content):
                emit(
                    "<turn|>\n",
                    msg_idx,
                    is_sampled=is_assistant,
                    is_content=is_assistant,
                )

        if add_generation_prompt and prev_message_type not in {
            "tool_response",
            "tool_call",
        }:
            emit("<|turn>model\n", -1, is_sampled=False, is_content=False)
            if (
                self._add_empty_thought_generation_prompt
                and not self.config.enable_thinking
            ):
                emit(
                    "<|channel>thought\n<channel|>",
                    -1,
                    is_sampled=False,
                    is_content=False,
                )

        return segments

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")
        segments = self._build_segments(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            include_bos=True,
            include_initial_block=True,
        )
        return self._segments_to_rendered(
            segments,
            message_roles=[m.get("role") or "" for m in messages],
            message_tool_names=extract_message_tool_names(messages),
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
        tools: list[ToolSpec] | None = None,  # noqa: ARG002 - Gemma4 DSL is self-describing.
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
        if (
            not previous_prompt_ids
            or not new_messages
            or self._uses_tooling(new_messages, tools)
            or any(msg.get("role") == "assistant" for msg in new_messages)
            or should_rerender_for_thinking_retention(
                self.effective_thinking_retention, new_messages
            )
        ):
            return None

        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._turn_end, self._eos},
            synthesize_close=self._turn_end,
        )
        if previous_ids is None:
            return None

        bridge_segments = [
            _Segment("\n", -1, is_sampled=False, is_content=False),
            *self._build_segments(
                new_messages,
                tools=None,
                add_generation_prompt=True,
                include_bos=False,
                include_initial_block=False,
            ),
        ]
        bridge = self._segments_to_rendered(
            bridge_segments,
            message_roles=[m.get("role") or "" for m in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
        )
        total_len = len(previous_ids) + len(bridge.token_ids)
        return RenderedTokens(
            token_ids=previous_ids + bridge.token_ids,
            message_indices=[-1] * len(previous_ids) + bridge.message_indices,
            sampled_mask=[False] * total_len,
            is_content=[False] * len(previous_ids) + bridge.is_content,
            message_roles=bridge.message_roles,
            message_tool_names=bridge.message_tool_names,
        )
