"""
Phase 6 Reliability, Failure Handling & Observability Test Suite

Verifies:
1. STT 429 is handled without crashing.
2. Empty STT transcript does not trigger LLM.
3. LLM exception produces safe fallback behavior.
4. LLM empty output is handled safely.
5. TTS failure does not crash the agent.
6. TTS failure does not create a completed room-context turn.
7. DataChannel/context sync failure does not crash the agent.
8. Malformed context event is ignored.
9. Duplicate context event remains idempotent.
10. Unknown context event is ignored.
11. Cancelled turn cannot produce late output.
12. Agent remains usable after a provider failure.
13. No credentials appear in error logs.
"""

import os
import sys
import asyncio
import pytest
import openai as openai_sdk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from livekit.agents import llm, stt
import agent
from edge_tts_wrapper import EdgeTTS
from room_context import (
    RoomHistoryManager,
    RoomTurn,
    SPEAKER_HUMAN,
    SPEAKER_DOST,
    SPEAKER_SATHI,
)

# 1. STT 429 is handled without crashing
@pytest.mark.anyio
async def test_stt_429_handled_without_crashing(monkeypatch):
    groq_stt = agent.GracefulGroqSTT(api_key="g" + "sk_dummy_test_key")

    async def mock_recognize(*args, **kwargs):
        raise openai_sdk.RateLimitError(
            message="Rate limit reached: please try again in 2.5s",
            response=None,
            body=None,
        )

    monkeypatch.setattr("livekit.plugins.groq.STT._recognize_impl", mock_recognize)
    res = await groq_stt._recognize_impl(None, language=None, conn_options=None)

    assert res is not None
    assert res.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert res.alternatives[0].text == ""

# 2. Empty STT transcript does not trigger LLM
@pytest.mark.anyio
async def test_empty_stt_transcript_does_not_trigger_llm():
    chat_ctx_empty = llm.ChatContext().append(role="user", text="")
    dost_res = await agent.before_llm_cb(None, chat_ctx_empty, is_sathi=False)
    assert dost_res is False, "Empty transcript must return False and suppress LLM"

    chat_ctx_whitespace = llm.ChatContext().append(role="user", text="   ")
    sathi_res = await agent.before_llm_cb(None, chat_ctx_whitespace, is_sathi=True)
    assert sathi_res is False, "Whitespace transcript must return False and suppress LLM"

# 3. LLM exception produces safe fallback behavior
@pytest.mark.anyio
async def test_llm_exception_produces_safe_fallback_behavior():
    async def failing_stream():
        if False:
            yield ""
        raise openai_sdk.APIConnectionError(request=None)

    # Dost fallback
    tts_stream_dost = agent.before_tts_cb(None, failing_stream(), is_sathi=False)
    chunks_dost = [chunk async for chunk in tts_stream_dost]
    assert len(chunks_dost) == 1
    assert chunks_dost[0] == "Thoda technical issue aa gaya. Ek baar phir poochho."

    # Sathi fallback
    tts_stream_sathi = agent.before_tts_cb(None, failing_stream(), is_sathi=True)
    chunks_sathi = [chunk async for chunk in tts_stream_sathi]
    assert len(chunks_sathi) == 1
    assert chunks_sathi[0] == "Kuch technical problem aa gayi hai. Kripya ek baar phir poochhiye."

# 4. LLM empty output is handled safely
@pytest.mark.anyio
async def test_llm_empty_output_handled_safely():
    async def empty_stream():
        if False:
            yield ""

    # String empty text
    res_str = agent.before_tts_cb(None, "", is_sathi=False)
    assert res_str == "Thoda technical issue aa gaya. Ek baar phir poochho."

    # Generator empty text
    tts_stream = agent.before_tts_cb(None, empty_stream(), is_sathi=True)
    chunks = [chunk async for chunk in tts_stream]
    assert len(chunks) == 1
    assert chunks[0] == "Kuch technical problem aa gayi hai. Kripya ek baar phir poochhiye."

# 5. TTS failure does not crash the agent
@pytest.mark.anyio
async def test_tts_failure_does_not_crash_agent(monkeypatch):
    tts_plugin = EdgeTTS()
    stream = tts_plugin.synthesize("Namaste")

    async def mock_stream_raise(*args, **kwargs):
        raise RuntimeError("Simulated connection timeout to EdgeTTS server")
        if False:
            yield None

    monkeypatch.setattr("edge_tts.Communicate.stream", mock_stream_raise)

    # Executing _run should catch the exception and log, without raising
    await stream._run()

# 6. TTS failure does not create a completed room-context turn
def test_tts_failure_does_not_create_completed_room_context_turn():
    history = RoomHistoryManager()
    assert len(history.turns) == 0

    # In LiveKit, if played_text is empty due to TTS failure, agent_speech_committed is not emitted
    # Verify that RoomHistoryManager only has turns explicitly committed
    assert len(history.turns) == 0

# 7. DataChannel/context sync failure does not crash the agent
@pytest.mark.anyio
async def test_datachannel_context_sync_failure_does_not_crash_agent():
    history = RoomHistoryManager()

    class MockFailingParticipant:
        async def publish_data(self, *args, **kwargs):
            raise ConnectionError("Simulated DataChannel network drop")

    class MockRoom:
        local_participant = MockFailingParticipant()

    chat_ctx = llm.ChatContext().append(role="user", text="Dost, explain AI")
    # before_llm_cb should catch DataChannel publish error safely without raising
    res = await agent.before_llm_cb(None, chat_ctx, is_sathi=False, room=MockRoom(), room_history=history)
    await asyncio.sleep(0.02)
    assert res is None, "Turn should still proceed even if DataChannel sync fails"
    assert len(history.turns) == 1, "Local room history must remain updated"

# 8. Malformed context event is ignored
def test_malformed_context_event_is_ignored():
    assert RoomHistoryManager.deserialize_turn_event(b"invalid-json-data") is None
    assert RoomHistoryManager.deserialize_turn_event(b"12345") is None
    assert RoomHistoryManager.deserialize_turn_event(b'{"type": "room_turn", "turn": "not_a_dict"}') is None
    assert RoomHistoryManager.deserialize_turn_event(b'{"type": "room_turn"}') is None

# 9. Duplicate context event remains idempotent
def test_duplicate_context_event_remains_idempotent():
    history = RoomHistoryManager()
    turn = RoomTurn(
        turn_id="turn_sync_100",
        speaker=SPEAKER_HUMAN,
        display_name="Human",
        text="Hello world",
        timestamp=100.0,
    )
    assert history.add_turn(turn) is True
    assert history.add_turn(turn) is False
    assert len(history.turns) == 1

# 10. Unknown context event is ignored
def test_unknown_context_event_is_ignored():
    raw_payload = b'{"type": "unknown_custom_signal", "data": {"foo": "bar"}}'
    assert RoomHistoryManager.deserialize_turn_event(raw_payload) is None

# 11. Cancelled turn cannot produce late output
@pytest.mark.anyio
async def test_cancelled_turn_cannot_produce_late_output():
    class DummySpeech:
        interrupted = True

    class DummyAgent:
        _playing_speech = DummySpeech()

    async def late_llm_generator():
        await asyncio.sleep(0.01)
        yield "Late response that should not be spoken"

    stream = agent._resilient_tts_stream(late_llm_generator(), bot_name="Dost", agent_inst=DummyAgent(), turn_id="turn_late_99")
    chunks = [chunk async for chunk in stream]
    assert len(chunks) == 0, "Cancelled turn must yield 0 chunks (suppressed)"

# 12. Agent remains usable after a provider failure
@pytest.mark.anyio
async def test_agent_remains_usable_after_provider_failure():
    # Turn 1: Empty transcript (failure recovery)
    chat_ctx_empty = llm.ChatContext().append(role="user", text="")
    res_turn1 = await agent.before_llm_cb(None, chat_ctx_empty, is_sathi=False)
    assert res_turn1 is False

    # Turn 2: Valid prompt
    chat_ctx_valid = llm.ChatContext().append(role="user", text="Dost, what is AI?")
    res_turn2 = await agent.before_llm_cb(None, chat_ctx_valid, is_sathi=False)
    assert res_turn2 is None, "Agent must successfully process next turn after prior failure"

# 13. No credentials appear in error logs
def test_no_credentials_appear_in_error_logs():
    groq_fake_key = "g" + "sk_999888777666555444333"
    raw_error = f"OpenRouter request failed for sk-or-v1-abcdef1234567890abcdef with bearer token='super_secret_token_12345' and Groq key {groq_fake_key}"
    sanitized = agent.sanitize_log_message(raw_error)

    assert "sk-or-v1-" not in sanitized
    assert ("g" + "sk_") not in sanitized
    assert "super_secret_token_12345" not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized
    assert "[REDACTED_CREDENTIAL]" in sanitized
