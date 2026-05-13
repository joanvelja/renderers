"""Append-only rendered state for multi-turn generation streams.

The unit this module owns is a single model-sampled transcript: prior prompt
tokens, prior completion tokens, and the messages those tokens commit to. A
multi-agent environment is just a mapping from stream id to that transcript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Sequence

from renderers.base import (
    Message,
    MultiModalData,
    ParsedResponse,
    RenderedTokens,
    Renderer,
    ToolSpec,
    is_multimodal,
)


class StreamDivergence(ValueError):
    """Raised when new messages rewrite an already-committed stream prefix."""

    def __init__(
        self,
        stream_id: Hashable,
        *,
        committed_count: int,
        observed_count: int,
    ):
        self.stream_id = stream_id
        self.committed_count = committed_count
        self.observed_count = observed_count
        super().__init__(
            f"{stream_id}: messages do not extend the committed stream "
            f"({committed_count=} {observed_count=})"
        )


class StreamBridgeUnavailable(RuntimeError):
    """Raised when a renderer cannot extend an already-committed stream."""

    def __init__(self, stream_id: Hashable):
        self.stream_id = stream_id
        super().__init__(f"{stream_id}: renderer could not bridge committed stream")


@dataclass(frozen=True)
class RenderedStream:
    """Committed token state for one append-only generation stream."""

    messages: tuple[Message, ...] = field(default_factory=tuple)
    prompt_ids: tuple[int, ...] = field(default_factory=tuple)
    completion_ids: tuple[int, ...] = field(default_factory=tuple)
    completion_logprobs: tuple[float, ...] = field(default_factory=tuple)
    prompt_message_indices: tuple[int, ...] = field(default_factory=tuple)
    multi_modal_data: MultiModalData | None = None
    parsed_completion: ParsedResponse | None = None

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self.prompt_ids + self.completion_ids


@dataclass(frozen=True)
class PreparedTurn:
    """Prompt prepared for the next model call on a stream."""

    messages: tuple[Message, ...]
    prompt_ids: tuple[int, ...]
    message_indices: tuple[int, ...] = field(default_factory=tuple)
    multi_modal_data: MultiModalData | None = None
    previous_token_count: int = 0
    bridge_used: bool = False
    exact_continuation: bool = False


@dataclass(frozen=True)
class StreamSet:
    """Immutable mapping of stream ids to rendered generation state."""

    streams: dict[Hashable, RenderedStream] = field(default_factory=dict)

    def get(self, stream_id: Hashable) -> RenderedStream | None:
        return self.streams.get(stream_id)

    def prepare_append(
        self,
        stream_id: Hashable,
        messages: Sequence[Message],
        renderer: Renderer,
        *,
        tools: list[ToolSpec] | None = None,
    ) -> PreparedTurn:
        """Prepare prompt ids for ``messages`` extending ``stream_id``.

        If the stream has prior state, this uses the renderer's bridge so
        sampled assistant tokens stay byte-exact. When the bridge declines, the
        stream cannot be safely continued as token state.
        """

        next_messages = _snapshot_messages(messages)
        previous = self.streams.get(stream_id)
        if previous is None:
            return _prepare_from_render(renderer.render(
                list(next_messages),
                tools=tools,
                add_generation_prompt=True,
            ), next_messages)

        committed_count = len(previous.messages)
        if next_messages[:committed_count] != previous.messages:
            raise StreamDivergence(
                stream_id,
                committed_count=committed_count,
                observed_count=len(next_messages),
            )

        new_messages = list(next_messages[committed_count:])
        bridged = _bridge(renderer, previous, new_messages, tools=tools)
        if bridged is None:
            raise StreamBridgeUnavailable(stream_id)

        prompt_ids, message_indices, multi_modal_data = _coerce_rendered(bridged)
        return PreparedTurn(
            messages=next_messages,
            prompt_ids=prompt_ids,
            message_indices=message_indices,
            multi_modal_data=multi_modal_data,
            previous_token_count=len(previous.token_ids),
            bridge_used=True,
            exact_continuation=True,
        )

    def commit(
        self,
        stream_id: Hashable,
        prepared: PreparedTurn,
        *,
        completion_ids: Sequence[int],
        assistant_message: Message,
        completion_logprobs: Sequence[float] | None = None,
        parsed_completion: ParsedResponse | None = None,
    ) -> "StreamSet":
        """Return a new set with ``prepared`` and its assistant completion committed."""

        next_streams = dict(self.streams)
        next_streams[stream_id] = RenderedStream(
            messages=prepared.messages + _snapshot_messages([assistant_message]),
            prompt_ids=prepared.prompt_ids,
            completion_ids=tuple(completion_ids),
            completion_logprobs=tuple(completion_logprobs or ()),
            prompt_message_indices=prepared.message_indices,
            multi_modal_data=prepared.multi_modal_data,
            parsed_completion=parsed_completion,
        )
        return StreamSet(next_streams)


def _bridge(
    renderer: Renderer,
    previous: RenderedStream,
    new_messages: list[Message],
    *,
    tools: list[ToolSpec] | None,
) -> RenderedTokens | None:
    if is_multimodal(renderer):
        return renderer.bridge_to_next_turn(
            list(previous.prompt_ids),
            list(previous.completion_ids),
            new_messages,
            tools=tools,
            previous_multi_modal_data=previous.multi_modal_data,
        )
    return renderer.bridge_to_next_turn(
        list(previous.prompt_ids),
        list(previous.completion_ids),
        new_messages,
        tools=tools,
    )


def _prepare_from_render(
    rendered: RenderedTokens,
    messages: tuple[Message, ...],
) -> PreparedTurn:
    prompt_ids, message_indices, multi_modal_data = _coerce_rendered(rendered)
    return PreparedTurn(
        messages=messages,
        prompt_ids=prompt_ids,
        message_indices=message_indices,
        multi_modal_data=multi_modal_data,
    )


def _coerce_rendered(
    rendered: RenderedTokens,
) -> tuple[tuple[int, ...], tuple[int, ...], MultiModalData | None]:
    return (
        tuple(rendered.token_ids),
        tuple(rendered.message_indices),
        rendered.multi_modal_data,
    )


def _snapshot_messages(messages: Sequence[Message]) -> tuple[Message, ...]:
    return tuple(dict(message) for message in messages)


__all__ = [
    "PreparedTurn",
    "RenderedStream",
    "StreamBridgeUnavailable",
    "StreamDivergence",
    "StreamSet",
]
