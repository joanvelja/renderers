from renderers.base import (
    MultiModalData,
    ParsedResponse,
    ParsedToolCall,
    RenderedTokens,
    ToolCallParseStatus,
)
from renderers.streams import (
    CompletedResponse,
    PreparedTurn,
    StreamBridgeUnavailable,
    StreamDivergence,
    StreamSet,
)


class _FakeRenderer:
    def __init__(self, *, bridge_result="tokens"):
        self.bridge_result = bridge_result
        self.render_calls = []
        self.bridge_calls = []

    def render(self, messages, *, tools=None, add_generation_prompt=False):
        self.render_calls.append((messages, tools, add_generation_prompt))
        token_ids = _encode_messages(messages)
        if add_generation_prompt:
            token_ids.append(900)
        return RenderedTokens(
            token_ids=token_ids,
            message_indices=_message_indices(messages, token_ids),
        )

    def bridge_to_next_turn(
        self,
        previous_prompt_ids,
        previous_completion_ids,
        new_messages,
        *,
        tools=None,
    ):
        self.bridge_calls.append(
            (previous_prompt_ids, previous_completion_ids, new_messages, tools)
        )
        if self.bridge_result is None:
            return None

        token_ids = (
            list(previous_prompt_ids)
            + list(previous_completion_ids)
            + _encode_messages(new_messages)
            + [900]
        )
        return RenderedTokens(
            token_ids=token_ids,
            message_indices=[-1] * (len(previous_prompt_ids) + len(previous_completion_ids))
            + _message_indices(new_messages, token_ids),
        )


def _encode_messages(messages):
    ids = []
    for message in messages:
        role_id = {"system": 10, "user": 20, "assistant": 30}.get(
            message["role"], 40
        )
        ids.extend([role_id, len(str(message.get("content", "")))])
    return ids


def _message_indices(messages, token_ids):
    indices = []
    for i, message in enumerate(messages):
        indices.extend([i, i])
    if len(indices) < len(token_ids):
        indices.extend([-1] * (len(token_ids) - len(indices)))
    return indices


def test_prepare_first_turn_renders_without_mutating_stream_set():
    renderer = _FakeRenderer()
    streams = StreamSet()

    prepared = streams.prepare_append(
        "prover",
        [{"role": "user", "content": "solve"}],
        renderer,
    )

    assert prepared.prompt_ids == (20, 5, 900)
    assert prepared.previous_token_count == 0
    assert not prepared.bridge_used
    assert not prepared.exact_continuation
    assert streams.get("prover") is None


def test_commit_returns_new_stream_set_with_completion_tape():
    renderer = _FakeRenderer()
    streams = StreamSet()
    prepared = streams.prepare_append(
        "prover",
        [{"role": "user", "content": "solve"}],
        renderer,
    )
    parsed = ParsedResponse(content="answer")

    committed = streams.commit(
        "prover",
        prepared,
        completion_ids=[7, 8],
        assistant_message={"role": "assistant", "content": "answer"},
        completion_logprobs=[-0.1, -0.2],
        parsed_completion=parsed,
    )

    assert streams.get("prover") is None
    stream = committed.get("prover")
    assert stream is not None
    assert stream.token_ids == (20, 5, 900, 7, 8)
    assert stream.completion_logprobs == (-0.1, -0.2)
    assert stream.parsed_completion is parsed
    assert stream.messages == (
        {"role": "user", "content": "solve"},
        {"role": "assistant", "content": "answer"},
    )


def test_commit_response_builds_assistant_message_from_parsed_response():
    prepared = StreamSet().prepare_append(
        "prover",
        [{"role": "user", "content": "solve"}],
        _FakeRenderer(),
    )
    parsed = ParsedResponse(
        content="answer",
        reasoning_content="work",
        tool_calls=[
            ParsedToolCall(
                raw='{"name": "lookup", "arguments": {"x": 1}}',
                name="lookup",
                arguments={"x": 1},
                status=ToolCallParseStatus.OK,
                id="call_1",
            ),
            ParsedToolCall(
                raw="{bad json",
                status=ToolCallParseStatus.INVALID_JSON,
            ),
        ],
    )

    streams = StreamSet().commit_response(
        "prover",
        prepared,
        CompletedResponse(
            completion_ids=[7, 8],
            completion_logprobs=[-0.1, -0.2],
            parsed=parsed,
        ),
    )

    stream = streams.get("prover")
    assert stream is not None
    assert stream.parsed_completion is parsed
    assert stream.messages[-1] == {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "work",
        "tool_calls": [
            {
                "type": "function",
                "id": "call_1",
                "function": {"name": "lookup", "arguments": {"x": 1}},
            }
        ],
    }


def test_commit_response_preserves_prompt_state_from_prepared_turn():
    multi_modal_data = MultiModalData(mm_hashes={"image": ["hash"]})
    prepared = PreparedTurn(
        messages=({"role": "user", "content": "solve"},),
        prompt_ids=(1, 2, 3),
        message_indices=(0, 0, -1),
        multi_modal_data=multi_modal_data,
    )

    streams = StreamSet().commit_response(
        "prover",
        prepared,
        CompletedResponse(
            completion_ids=[4],
            completion_logprobs=[-0.3],
            parsed=ParsedResponse(content="answer"),
        ),
    )

    stream = streams.get("prover")
    assert stream is not None
    assert stream.prompt_ids == (1, 2, 3)
    assert stream.prompt_message_indices == (0, 0, -1)
    assert stream.multi_modal_data is multi_modal_data


def test_commit_response_rejects_logprob_length_mismatch():
    prepared = StreamSet().prepare_append(
        "prover",
        [{"role": "user", "content": "solve"}],
        _FakeRenderer(),
    )

    try:
        StreamSet().commit_response(
            "prover",
            prepared,
            CompletedResponse(
                completion_ids=[7, 8],
                completion_logprobs=[-0.1],
                parsed=ParsedResponse(content="answer"),
            ),
        )
    except ValueError as exc:
        assert "completion_logprobs" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_two_streams_bridge_independently_from_shared_prefix():
    renderer = _FakeRenderer()
    streams = StreamSet()

    for member_id, answer_id in [("prover", 101), ("verifier", 202)]:
        prepared = streams.prepare_append(
            member_id,
            [{"role": "user", "content": "solve"}],
            renderer,
        )
        streams = streams.commit(
            member_id,
            prepared,
            completion_ids=[answer_id],
            assistant_message={"role": "assistant", "content": f"a{answer_id}"},
        )

    prover_next = streams.prepare_append(
        "prover",
        [
            {"role": "user", "content": "solve"},
            {"role": "assistant", "content": "a101"},
            {"role": "user", "content": "respond"},
        ],
        renderer,
    )
    verifier_next = streams.prepare_append(
        "verifier",
        [
            {"role": "user", "content": "solve"},
            {"role": "assistant", "content": "a202"},
            {"role": "user", "content": "respond"},
        ],
        renderer,
    )

    assert prover_next.bridge_used
    assert verifier_next.bridge_used
    assert prover_next.prompt_ids[: prover_next.previous_token_count] == (
        20,
        5,
        900,
        101,
    )
    assert verifier_next.prompt_ids[: verifier_next.previous_token_count] == (
        20,
        5,
        900,
        202,
    )
    assert len(renderer.render_calls) == 2
    assert len(renderer.bridge_calls) == 2


def test_prepare_rejects_rewritten_stream_history():
    renderer = _FakeRenderer()
    prepared = StreamSet().prepare_append(
        "prover",
        [{"role": "user", "content": "solve"}],
        renderer,
    )
    streams = StreamSet().commit(
        "prover",
        prepared,
        completion_ids=[101],
        assistant_message={"role": "assistant", "content": "answer"},
    )

    try:
        streams.prepare_append(
            "prover",
            [
                {"role": "user", "content": "different"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "respond"},
            ],
            renderer,
        )
    except StreamDivergence as exc:
        assert exc.stream_id == "prover"
    else:
        raise AssertionError("expected StreamDivergence")


def test_prepare_fails_when_bridge_returns_none():
    renderer = _FakeRenderer(bridge_result=None)
    prepared = StreamSet().prepare_append(
        "prover",
        [{"role": "user", "content": "solve"}],
        renderer,
    )
    streams = StreamSet().commit(
        "prover",
        prepared,
        completion_ids=[101],
        assistant_message={"role": "assistant", "content": "answer"},
    )

    try:
        streams.prepare_append(
            "prover",
            [
                {"role": "user", "content": "solve"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "respond"},
            ],
            renderer,
        )
    except StreamBridgeUnavailable as exc:
        assert exc.stream_id == "prover"
    else:
        raise AssertionError("expected StreamBridgeUnavailable")

    assert len(renderer.render_calls) == 1
    assert len(renderer.bridge_calls) == 1
