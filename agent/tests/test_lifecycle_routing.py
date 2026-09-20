import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from livekit.agents import llm
from router import route_turn
import agent

@pytest.mark.anyio
async def test_sathi_selected_suppresses_dost_llm():
    """Proves: Sathi selected -> Dost does NOT invoke LLM (returns False), Sathi returns None."""
    chat_ctx_dost = llm.ChatContext().append(role="user", text="Sathi, AI kya hota hai?")
    dost_res = await agent.before_llm_cb(None, chat_ctx_dost, is_sathi=False)
    assert dost_res is False, "Dost must return False to suppress LLM when Sathi is addressed."

    chat_ctx_sathi = llm.ChatContext().append(role="user", text="Sathi, AI kya hota hai?")
    sathi_res = await agent.before_llm_cb(None, chat_ctx_sathi, is_sathi=True)
    assert sathi_res is None, "Sathi must return None to allow LLM synthesis."

@pytest.mark.anyio
async def test_dost_selected_suppresses_sathi_llm():
    """Proves: Dost selected -> Sathi does NOT invoke LLM (returns False), Dost returns None."""
    chat_ctx_sathi = llm.ChatContext().append(role="user", text="Dost, cloud computing kya hoti hai?")
    sathi_res = await agent.before_llm_cb(None, chat_ctx_sathi, is_sathi=True)
    assert sathi_res is False, "Sathi must return False to suppress LLM when Dost is addressed."

    chat_ctx_dost = llm.ChatContext().append(role="user", text="Dost, cloud computing kya hoti hai?")
    dost_res = await agent.before_llm_cb(None, chat_ctx_dost, is_sathi=False)
    assert dost_res is None, "Dost must return None to allow LLM synthesis."

@pytest.mark.anyio
async def test_default_unaddressed_selects_dost_only():
    """Proves: Unaddressed default question -> Only Dost invokes LLM."""
    chat_ctx_dost = llm.ChatContext().append(role="user", text="AI kya hota hai?")
    dost_res = await agent.before_llm_cb(None, chat_ctx_dost, is_sathi=False)
    assert dost_res is None, "Default question must allow Dost to respond."

    chat_ctx_sathi = llm.ChatContext().append(role="user", text="AI kya hota hai?")
    sathi_res = await agent.before_llm_cb(None, chat_ctx_sathi, is_sathi=True)
    assert sathi_res is False, "Default question must suppress Sathi."

@pytest.mark.anyio
async def test_empty_or_continue_prompt_suppresses_both():
    """Proves: Empty or <continue> prompts return False for before_llm_cb."""
    chat_ctx_empty = llm.ChatContext().append(role="user", text="<continue>")
    dost_res = await agent.before_llm_cb(None, chat_ctx_empty, is_sathi=False)
    assert dost_res is False, "Dost must return False for empty/continue prompts."

    sathi_res = await agent.before_llm_cb(None, chat_ctx_empty, is_sathi=True)
    assert sathi_res is False, "Sathi must return False for empty/continue prompts."

@pytest.mark.anyio
async def test_gate2_tts_secondary_suppression():
    """Proves: Gate 2 before_tts_cb suppresses Dost speech if latest transcript updated to Sathi."""
    class DummyAgent:
        _transcribed_text = "Sathi, AI kya hota hai?"

    dummy_agent = DummyAgent()

    # Dost tries to run before_tts_cb with text chunk
    dost_tts_out = agent.before_tts_cb(dummy_agent, "Some text...", is_sathi=False)
    assert dost_tts_out == "", "Dost before_tts_cb must return empty string when latest transcript is owned by Sathi."

    # Sathi runs before_tts_cb with text chunk
    sathi_tts_out = agent.before_tts_cb(dummy_agent, "Some text...", is_sathi=True)
    assert sathi_tts_out == "Some text...", "Sathi before_tts_cb must return text when Sathi owns transcript."

@pytest.mark.anyio
async def test_both_selected_allows_both_sequentially():
    """Proves: both selected -> Order 1 allows LLM (None), Order 2 suppresses immediate LLM (False)."""
    chat_ctx_dost = llm.ChatContext().append(role="user", text="Dost aur Sathi, dono AI ke baare mein batao")
    dost_res = await agent.before_llm_cb(None, chat_ctx_dost, is_sathi=False)
    assert dost_res is None, "Dost (order 1) must return None when both are selected."

    class MockRoom:
        remote_participants = {}

    sathi_res = await agent.before_llm_cb(None, chat_ctx_dost, is_sathi=True, room=MockRoom())
    assert sathi_res is False, "Sathi (order 2) must return False to suppress immediate LLM until order 1 completes."

    # Reverse order test: Sathi first, Dost second
    chat_ctx_rev = llm.ChatContext().append(role="user", text="Sathi, pehle answer karo. Dost, baad mein example dena.")
    sathi_res_rev = await agent.before_llm_cb(None, chat_ctx_rev, is_sathi=True)
    assert sathi_res_rev is None, "Sathi (order 1) must return None when ordered first."

    dost_res_rev = await agent.before_llm_cb(None, chat_ctx_rev, is_sathi=False)
    assert dost_res_rev is False, "Dost (order 2) must return False when ordered second."

@pytest.mark.anyio
async def test_explicit_sathi_no_second_path_dost_trigger():
    """Proves: Data channel text chat route check suppresses Dost when Sathi is addressed."""
    text = "Sathi, text chat question"
    selected_agent = route_turn(text)
    assert selected_agent == "sathi"

    dost_triggered = (selected_agent in ("dost", "both"))
    sathi_triggered = (selected_agent in ("sathi", "both"))

    assert dost_triggered is False, "Dost data channel trigger must be False when Sathi is addressed."
    assert sathi_triggered is True, "Sathi data channel trigger must be True when Sathi is addressed."

@pytest.mark.anyio
async def test_devanagari_sathi_suppresses_dost():
    """Proves: Devanagari Sathi addressing suppresses Dost LLM."""
    chat_ctx = llm.ChatContext().append(role="user", text="साथी, AI क्या होता है?")
    dost_res = await agent.before_llm_cb(None, chat_ctx, is_sathi=False)
    assert dost_res is False, "Devanagari Sathi prompt must suppress Dost."

    sathi_res = await agent.before_llm_cb(None, chat_ctx, is_sathi=True)
    assert sathi_res is None, "Devanagari Sathi prompt must allow Sathi."
