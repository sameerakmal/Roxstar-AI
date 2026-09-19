"""
Phase 4 Shared Room-Level Context Test Suite

Tests:
1. Cross-bot cloud computing follow-up (Test 1: Dost -> Sathi)
2. Cross-bot machine learning follow-up (Test 2: Sathi -> Dost)
3. Cross-bot speaker fact recall (Test 3: Dost -> Sathi)
4. Duplicate human transcript prevention (Unselected bot does not add human turn)
5. Duplicate turn_id idempotency prevention
6. Bot-turn synchronization across process buffers
7. Sliding window bounds enforcement
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from livekit.agents import llm
import agent
from room_context import (
    RoomHistoryManager,
    RoomTurn,
    SPEAKER_HUMAN,
    SPEAKER_DOST,
    SPEAKER_SATHI,
    DISPLAY_NAME_HUMAN,
    DISPLAY_NAME_DOST,
    DISPLAY_NAME_SATHI,
)

@pytest.mark.anyio
async def test_cross_bot_cloud_computing_followup():
    """
    Test 1:
    Human: "Dost, explain cloud computing."
    Dost responds.
    Human: "Sathi, iska ek real-life example do."
    Verify Sathi receives Dost's response in context and resolves 'iska' to cloud computing.
    """
    history_dost = RoomHistoryManager(max_turns=20)
    history_sathi = RoomHistoryManager(max_turns=20)

    # Turn 1: Human addresses Dost
    chat_ctx_1 = llm.ChatContext().append(role="user", text="Dost, explain cloud computing.")
    dost_res = await agent.before_llm_cb(None, chat_ctx_1, is_sathi=False, room_history=history_dost)
    assert dost_res is None, "Dost should handle Turn 1"

    # Dost generates response and commits bot turn
    dost_turn = history_dost.create_bot_turn(
        speaker=SPEAKER_DOST,
        text="Cloud computing ek computer services over internet network hai jaise storage aur servers."
    )
    history_dost.add_turn(dost_turn)

    # DataChannel sync from Dost process to Sathi process
    for turn in history_dost.turns:
        history_sathi.add_turn(turn)

    # Turn 2: Human addresses Sathi with follow-up 'iska'
    chat_ctx_2 = llm.ChatContext().append(role="user", text="Sathi, iska ek real-life example do.")
    sathi_res = await agent.before_llm_cb(None, chat_ctx_2, is_sathi=True, room_history=history_sathi)
    assert sathi_res is None, "Sathi should handle Turn 2"

    # Verify Sathi's context contains both turns and Dost's output
    sathi_messages = chat_ctx_2.messages
    roles = [msg.role for msg in sathi_messages]
    contents = [msg.content for msg in sathi_messages]

    assert roles[0] == "system"
    assert f"[{DISPLAY_NAME_HUMAN}]: Dost, explain cloud computing." in contents[1]
    assert f"[{DISPLAY_NAME_DOST}]: Cloud computing ek computer services" in contents[2]
    assert f"[{DISPLAY_NAME_HUMAN}]: Sathi, iska ek real-life example do." in contents[3]

@pytest.mark.anyio
async def test_cross_bot_machine_learning_followup():
    """
    Test 2:
    Human: "Sathi, explain machine learning."
    Sathi responds.
    Human: "Dost, iska ek simple example do."
    Verify Dost receives Sathi's response in context and resolves 'iska' to machine learning.
    """
    history_dost = RoomHistoryManager(max_turns=20)
    history_sathi = RoomHistoryManager(max_turns=20)

    # Turn 1: Human addresses Sathi
    chat_ctx_1 = llm.ChatContext().append(role="user", text="Sathi, explain machine learning.")
    sathi_res = await agent.before_llm_cb(None, chat_ctx_1, is_sathi=True, room_history=history_sathi)
    assert sathi_res is None, "Sathi should handle Turn 1"

    # Sathi commits bot turn
    sathi_turn = history_sathi.create_bot_turn(
        speaker=SPEAKER_SATHI,
        text="Machine learning AI ki branch hai jo data patterns seekh kar automatic predictions karti hai."
    )
    history_sathi.add_turn(sathi_turn)

    # Sync to Dost process
    for turn in history_sathi.turns:
        history_dost.add_turn(turn)

    # Turn 2: Human addresses Dost with follow-up 'iska'
    chat_ctx_2 = llm.ChatContext().append(role="user", text="Dost, iska ek simple example do.")
    dost_res = await agent.before_llm_cb(None, chat_ctx_2, is_sathi=False, room_history=history_dost)
    assert dost_res is None, "Dost should handle Turn 2"

    # Verify Dost's context contains Sathi's explanation of machine learning
    dost_messages = chat_ctx_2.messages
    contents = [msg.content for msg in dost_messages]

    assert f"[{DISPLAY_NAME_HUMAN}]: Sathi, explain machine learning." in contents[1]
    assert f"[{DISPLAY_NAME_SATHI}]: Machine learning AI ki branch hai" in contents[2]
    assert f"[{DISPLAY_NAME_HUMAN}]: Dost, iska ek simple example do." in contents[3]

@pytest.mark.anyio
async def test_cross_bot_speaker_fact_recall():
    """
    Test 3:
    Human: "Dost, my favorite programming language is Python."
    Dost responds.
    Human: "Sathi, what programming language did I just mention?"
    Verify Sathi recalls 'Python' from shared context.
    """
    history_dost = RoomHistoryManager(max_turns=20)
    history_sathi = RoomHistoryManager(max_turns=20)

    chat_ctx_1 = llm.ChatContext().append(role="user", text="Dost, my favorite programming language is Python.")
    await agent.before_llm_cb(None, chat_ctx_1, is_sathi=False, room_history=history_dost)

    dost_turn = history_dost.create_bot_turn(
        speaker=SPEAKER_DOST,
        text="Waah! Python bahut clean aur powerful language hai."
    )
    history_dost.add_turn(dost_turn)

    # Sync to Sathi
    for turn in history_dost.turns:
        history_sathi.add_turn(turn)

    chat_ctx_2 = llm.ChatContext().append(role="user", text="Sathi, what programming language did I just mention?")
    await agent.before_llm_cb(None, chat_ctx_2, is_sathi=True, room_history=history_sathi)

    contents = [msg.content for msg in chat_ctx_2.messages]
    assert any("Python" in content for content in contents), "Sathi's context must contain 'Python' from previous turn."

@pytest.mark.anyio
async def test_duplicate_human_transcript_prevention():
    """
    Proves constraint 5: When parallel STT emissions happen on Dost & Sathi,
    only the selected bot commits the canonical human turn. Unselected bot does NOT.
    """
    history_dost = RoomHistoryManager(max_turns=20)
    history_sathi = RoomHistoryManager(max_turns=20)

    prompt = "Sathi, AI kya hota hai?"

    chat_ctx_dost = llm.ChatContext().append(role="user", text=prompt)
    chat_ctx_sathi = llm.ChatContext().append(role="user", text=prompt)

    # Dost executes before_llm_cb (unselected for Sathi prompt)
    res_dost = await agent.before_llm_cb(None, chat_ctx_dost, is_sathi=False, room_history=history_dost)
    assert res_dost is False, "Dost must return False (suppressed)"
    assert len(history_dost.turns) == 0, "Unselected Dost must NOT add human turn to history"

    # Sathi executes before_llm_cb (selected for Sathi prompt)
    res_sathi = await agent.before_llm_cb(None, chat_ctx_sathi, is_sathi=True, room_history=history_sathi)
    assert res_sathi is None, "Sathi must return None (allowed)"
    assert len(history_sathi.turns) == 1, "Selected Sathi must add canonical human turn"
    assert history_sathi.turns[0].speaker == SPEAKER_HUMAN
    assert history_sathi.turns[0].text == prompt

@pytest.mark.anyio
async def test_duplicate_turn_id_prevention():
    """
    Proves constraint 6 & 7: Every turn has a unique turn_id and updates are idempotent.
    Duplicate turn_id is ignored.
    """
    history = RoomHistoryManager(max_turns=20)

    turn1 = RoomTurn(
        turn_id="unique_turn_001",
        speaker=SPEAKER_HUMAN,
        display_name=DISPLAY_NAME_HUMAN,
        text="Explain cloud computing",
        timestamp=1000.0,
    )

    added_1 = history.add_turn(turn1)
    assert added_1 is True, "First addition must return True"
    assert len(history.turns) == 1

    # Attempt to add duplicate turn with same turn_id
    added_2 = history.add_turn(turn1)
    assert added_2 is False, "Duplicate turn_id addition must return False"
    assert len(history.turns) == 1

@pytest.mark.anyio
async def test_bot_turn_synchronization():
    """
    Proves constraint 8 & 10: Bot turn events serialize and deserialize correctly
    for DataChannel synchronization across processes.
    """
    history_sender = RoomHistoryManager(max_turns=20)
    history_receiver = RoomHistoryManager(max_turns=20)

    bot_turn = history_sender.create_bot_turn(
        speaker=SPEAKER_DOST,
        text="Cloud computing ek online storage platform hai."
    )
    history_sender.add_turn(bot_turn)

    # Serialize
    packet_bytes = history_sender.serialize_turn_event(bot_turn)
    assert isinstance(packet_bytes, bytes)

    # Deserialize
    deserialized_turn = RoomHistoryManager.deserialize_turn_event(packet_bytes)
    assert deserialized_turn is not None
    assert deserialized_turn.turn_id == bot_turn.turn_id
    assert deserialized_turn.speaker == SPEAKER_DOST
    assert deserialized_turn.text == "Cloud computing ek online storage platform hai."

    # Receive
    added = history_receiver.add_turn(deserialized_turn)
    assert added is True
    assert len(history_receiver.turns) == 1
    assert history_receiver.turns[0].speaker == SPEAKER_DOST

@pytest.mark.anyio
async def test_sliding_window_bounds():
    """
    Proves constraint 12: RoomHistoryManager respects max_turns sliding window limit.
    """
    history = RoomHistoryManager(max_turns=5)

    for i in range(8):
        t = history.create_human_turn(f"Utterance number {i}")
        history.add_turn(t)

    assert len(history.turns) == 5, "History length must be capped at max_turns (5)"
    assert history.turns[0].text == "Utterance number 3", "Oldest turns (0, 1, 2) must be evicted"
    assert history.turns[-1].text == "Utterance number 7"

def test_stored_dost_turn_contains_no_prefix():
    """Regression test: Stored Dost turn contains no '[Roxstar AI Dost]' prefix."""
    history = RoomHistoryManager()
    dirty_text = "[Roxstar AI Dost]: Simple example: Jab aap apne phone par backup lete ho"
    bot_turn = history.create_bot_turn(speaker=SPEAKER_DOST, text=dirty_text)
    history.add_turn(bot_turn)

    stored_text = history.turns[0].text
    assert not stored_text.startswith("[Roxstar AI Dost]"), "Stored Dost turn must not have [Roxstar AI Dost] prefix"
    assert not stored_text.startswith("Roxstar AI Dost"), "Stored Dost turn must not have Roxstar AI Dost prefix"
    assert stored_text == "Simple example: Jab aap apne phone par backup lete ho"

def test_stored_sathi_turn_contains_no_prefix():
    """Regression test: Stored Sathi turn contains no '[Roxstar AI Sathi]' prefix."""
    history = RoomHistoryManager()
    dirty_text = "[Roxstar AI Sathi]: Yeh machine learning ka real life example hai"
    bot_turn = history.create_bot_turn(speaker=SPEAKER_SATHI, text=dirty_text)
    history.add_turn(bot_turn)

    stored_text = history.turns[0].text
    assert not stored_text.startswith("[Roxstar AI Sathi]"), "Stored Sathi turn must not have [Roxstar AI Sathi] prefix"
    assert not stored_text.startswith("Roxstar AI Sathi"), "Stored Sathi turn must not have Roxstar AI Sathi prefix"
    assert stored_text == "Yeh machine learning ka real life example hai"

def test_formatted_llm_context_contains_labels_exactly_once():
    """Regression test: Formatted LLM context contains labels exactly once."""
    history = RoomHistoryManager()
    history.add_turn(history.create_human_turn("Explain cloud computing"))
    history.add_turn(history.create_bot_turn(SPEAKER_DOST, "Cloud computing is on-demand computing"))
    history.add_turn(history.create_human_turn("Iska ek example do"))
    history.add_turn(history.create_bot_turn(SPEAKER_SATHI, "Google Drive is an example"))

    ctx = history.build_chat_context(system_prompt="System Prompt")
    contents = [msg.content for msg in ctx.messages[1:]]

    # Verify each message has its label exactly once
    assert contents[0] == "[Human]: Explain cloud computing"
    assert contents[1] == "[Roxstar AI Dost]: Cloud computing is on-demand computing"
    assert contents[2] == "[Human]: Iska ek example do"
    assert contents[3] == "[Roxstar AI Sathi]: Google Drive is an example"

    for content in contents:
        assert content.count("[Human]") <= 1
        assert content.count("[Roxstar AI Dost]") <= 1
        assert content.count("[Roxstar AI Sathi]") <= 1
        assert "[Roxstar AI Dost]: [Roxstar AI Dost]" not in content
        assert "[Roxstar AI Sathi]: [Roxstar AI Sathi]" not in content

def test_bot_output_does_not_inherit_metadata_prefix():
    """Regression test: before_tts_cb strips metadata prefix from bot speech."""
    dirty_dost = "[Roxstar AI Dost]: Simple example: Jab aap phone use karte ho"
    cleaned_dost = agent.before_tts_cb(None, dirty_dost, is_sathi=False)
    assert cleaned_dost == "Simple example: Jab aap phone use karte ho"

    dirty_sathi = "[Roxstar AI Sathi]: Bilkul, main ek example deti hoon"
    cleaned_sathi = agent.before_tts_cb(None, dirty_sathi, is_sathi=True)
    assert cleaned_sathi == "Bilkul, main ek example deti hoon"

    # Also test unbracketed name prefix
    unbracketed = "Roxstar AI Dost: Main aapki madad karta hoon"
    assert agent.before_tts_cb(None, unbracketed, is_sathi=False) == "Main aapki madad karta hoon"

def test_repeated_multiturn_context_does_not_accumulate_speaker_labels():
    """Regression test: Repeated multi-turn context does not accumulate speaker labels."""
    history = RoomHistoryManager()

    # Simulate 3 consecutive exchanges where bot outputs might have accidentally carried labels
    history.add_turn(history.create_human_turn("Dost, cloud computing kya hai?"))
    history.add_turn(history.create_bot_turn(SPEAKER_DOST, "[Roxstar AI Dost]: Cloud computing ek internet service hai"))

    # Turn 2
    history.add_turn(history.create_human_turn("Sathi, iska example do"))
    history.add_turn(history.create_bot_turn(SPEAKER_SATHI, "[Roxstar AI Sathi]: Jaise Google Cloud Platform"))

    # Turn 3
    history.add_turn(history.create_human_turn("Dost, aur kuch?"))
    history.add_turn(history.create_bot_turn(SPEAKER_DOST, "[Roxstar AI Dost]: [Roxstar AI Dost]: Aur yeh scalable bhi hai"))

    # Check stored turns in history
    for turn in history.turns:
        assert not turn.text.startswith("[Roxstar AI")
        assert not turn.text.startswith("Roxstar AI")

    # Check formatted context
    ctx = history.build_chat_context("System")
    for msg in ctx.messages[1:]:
        assert "[Roxstar AI Dost]: [Roxstar AI Dost]" not in msg.content
        assert "[Roxstar AI Sathi]: [Roxstar AI Sathi]" not in msg.content
        assert msg.content.count("[Roxstar AI Dost]") <= 1
        assert msg.content.count("[Roxstar AI Sathi]") <= 1

