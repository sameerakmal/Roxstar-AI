import os
import sys
import json
import time
import pytest
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from router import (
    route_turn,
    is_explicit_address,
    parse_ordered_plan,
    MultiBotPlan,
    MultiBotTarget,
)
from room_context import RoomHistoryManager
from livekit.agents import llm
import agent
from agent import OrchestrationManager

# ── 1. Plan Parsing Tests ───────────────────────────────────────────────────

def test_parse_ordered_plan_dost_first():
    """Proves: 'AI Dost, tum answer karo. AI Sathi, baad mein ek example dena.' sets Dost=1, Sathi=2."""
    text = "AI Dost, tum answer karo. AI Sathi, baad mein ek example dena."
    plan = parse_ordered_plan(text)
    assert plan is not None
    assert plan.original_text == text
    assert len(plan.targets) == 2

    dost_target = next(t for t in plan.targets if t.bot == "dost")
    sathi_target = next(t for t in plan.targets if t.bot == "sathi")

    assert dost_target.order == 1
    assert sathi_target.order == 2
    assert "answer karo" in dost_target.instruction.lower()
    assert "example" in sathi_target.instruction.lower()


def test_parse_ordered_plan_sathi_first():
    """Proves: 'Sathi, pehle explain karo. Dost, baad mein summarize karna.' sets Sathi=1, Dost=2."""
    text = "Sathi, pehle explain karo. Dost, baad mein summarize karna."
    plan = parse_ordered_plan(text)
    assert plan is not None

    dost_target = next(t for t in plan.targets if t.bot == "dost")
    sathi_target = next(t for t in plan.targets if t.bot == "sathi")

    assert sathi_target.order == 1
    assert dost_target.order == 2
    assert "explain" in sathi_target.instruction.lower()
    assert "summarize" in dost_target.instruction.lower()


def test_parse_ordered_plan_positional_order():
    """Proves: Without explicit order keywords, positional occurrence order determines order."""
    text = "Dost aur Sathi, dono machine learning ke baare mein batao"
    plan = parse_ordered_plan(text)
    assert plan is not None

    dost_target = next(t for t in plan.targets if t.bot == "dost")
    sathi_target = next(t for t in plan.targets if t.bot == "sathi")

    assert dost_target.order == 1
    assert sathi_target.order == 2


def test_parse_ordered_plan_devanagari():
    """Proves: Devanagari script order cues ('पहले' and 'बाद में') are parsed accurately."""
    text = "दोस्त पहले समझाओ. साथी बाद में उदाहरण दो."
    plan = parse_ordered_plan(text)
    assert plan is not None

    dost_target = next(t for t in plan.targets if t.bot == "dost")
    sathi_target = next(t for t in plan.targets if t.bot == "sathi")

    assert dost_target.order == 1
    assert sathi_target.order == 2


def test_turn_id_determinism():
    """Proves: Both agents parsing identical user text generate identical turn_id."""
    text = "AI Dost, tum answer karo. AI Sathi, baad mein ek example dena."
    plan1 = parse_ordered_plan(text)
    plan2 = parse_ordered_plan(text)
    assert plan1.turn_id == plan2.turn_id
    assert plan1.turn_id.startswith("plan_")


# ── 2. Referential Mentions vs Explicit Address ─────────────────────────────

def test_referential_mention_sathi_addressed_dost_referenced():
    """Proves: 'Sathi, Dost ne jo bola uska example do' routes ONLY to Sathi (Dost is referential)."""
    text = "Sathi, Dost ne jo bola uska example do"
    assert is_explicit_address(text, "sathi") is True
    assert is_explicit_address(text, "dost") is False
    assert route_turn(text) == "sathi"


def test_referential_mention_dost_addressed_sathi_referenced():
    """Proves: 'Dost, Sathi ka example sahi tha kya?' routes ONLY to Dost (Sathi is referential)."""
    text = "Dost, Sathi ka example sahi tha kya?"
    assert is_explicit_address(text, "dost") is True
    assert is_explicit_address(text, "sathi") is False
    assert route_turn(text) == "dost"


def test_referential_mention_devanagari():
    """Proves: 'साथी, दोस्त ने जो बोला समझाओ' routes ONLY to Sathi."""
    text = "साथी, दोस्त ने जो बोला समझाओ"
    assert is_explicit_address(text, "sathi") is True
    assert is_explicit_address(text, "dost") is False
    assert route_turn(text) == "sathi"


# ── 3. Lifecycle before_llm_cb Coordination ─────────────────────────────────

@pytest.mark.anyio
async def test_before_llm_order1_allowed_order2_suppressed():
    """Proves: In ordered plan, Order 1 returns None (proceed) and Order 2 returns False (suppressed)."""
    orch_dost = OrchestrationManager()
    orch_sathi = OrchestrationManager()

    text = "AI Dost, tum answer karo. AI Sathi, baad mein ek example dena."
    chat_ctx = llm.ChatContext().append(role="user", text=text)

    # Dost is order 1
    dost_res = await agent.before_llm_cb(
        None, chat_ctx, is_sathi=False, orchestrator=orch_dost
    )
    assert dost_res is None, "Order 1 bot must return None to proceed with LLM."
    assert orch_dost.my_order == 1
    assert orch_dost.current_plan is not None

    # Sathi is order 2
    sathi_res = await agent.before_llm_cb(
        None, chat_ctx, is_sathi=True, orchestrator=orch_sathi
    )
    assert sathi_res is False, "Order 2 bot must return False to suppress immediate LLM turn."
    assert orch_sathi.my_order == 2
    assert orch_sathi.pending_order2_plan is not None


@pytest.mark.anyio
async def test_before_llm_order1_enrichment_injected():
    """Proves: Order 1 bot receives prompt enrichment focusing on its clause."""
    orch = OrchestrationManager()
    text = "AI Dost, tum answer karo. AI Sathi, baad mein ek example dena."
    chat_ctx = llm.ChatContext().append(role="user", text=text)

    await agent.before_llm_cb(None, chat_ctx, is_sathi=False, orchestrator=orch)

    # Check that system role enrichment was added to chat_ctx
    system_messages = [m for m in chat_ctx.messages if m.role == "system"]
    assert len(system_messages) > 0
    enrichment_found = any("Instruction for this turn" in m.content for m in system_messages)
    assert enrichment_found is True


@pytest.mark.anyio
async def test_before_llm_reverse_ordering():
    """Proves: When Sathi is first, Sathi gets None (proceed) and Dost gets False (suppressed)."""
    orch_dost = OrchestrationManager()
    orch_sathi = OrchestrationManager()

    text = "Sathi, pehle explain karo. Dost, baad mein summarize karna."
    chat_ctx = llm.ChatContext().append(role="user", text=text)

    sathi_res = await agent.before_llm_cb(None, chat_ctx, is_sathi=True, orchestrator=orch_sathi)
    assert sathi_res is None, "Sathi (order 1) must return None."
    assert orch_sathi.my_order == 1

    dost_res = await agent.before_llm_cb(None, chat_ctx, is_sathi=False, orchestrator=orch_dost)
    assert dost_res is False, "Dost (order 2) must return False."
    assert orch_dost.my_order == 2


# ── 4. OrchestrationManager State Transitions ────────────────────────────────

def test_orchestration_manager_reset():
    """Proves: OrchestrationManager.reset sets up new plan state cleanly."""
    mgr = OrchestrationManager()
    plan = parse_ordered_plan("Dost aur Sathi, dono batao")
    mgr.reset(plan, my_order=1)

    assert mgr.current_plan == plan
    assert mgr.my_order == 1
    assert mgr.is_interrupted is False


def test_orchestration_manager_mark_order1_finished():
    """Proves: mark_order1_finished clears active plan."""
    mgr = OrchestrationManager()
    plan = parse_ordered_plan("Dost aur Sathi, dono batao")
    mgr.reset(plan, my_order=1)
    mgr.mark_order1_finished()

    assert mgr.current_plan is None
    assert mgr.my_order == None


def test_orchestration_manager_cancel_plan():
    """Proves: cancel_current_plan clears both active and pending plans."""
    mgr = OrchestrationManager()
    plan = parse_ordered_plan("Dost aur Sathi, dono batao")
    mgr.pending_order2_plan = plan
    mgr.current_plan = plan
    mgr.cancel_current_plan()

    assert mgr.current_plan is None
    assert mgr.pending_order2_plan is None


# ── 5. Completion Signaling & Interruption Tests ──────────────────────────────

@pytest.mark.anyio
async def test_order2_execution_with_context_enrichment():
    """Proves: _execute_order2_response uses room history + per-bot instruction."""
    history = RoomHistoryManager()
    human_turn = history.create_human_turn("AI Dost, tum answer karo. AI Sathi, baad mein ek example dena.")
    history.add_turn(human_turn)
    dost_turn = history.create_bot_turn("Roxstar AI Dost", "AI ek technology hai jo machines ko smart banati hai.")
    history.add_turn(dost_turn)

    plan = parse_ordered_plan("AI Dost, tum answer karo. AI Sathi, baad mein ek example dena.")
    orch = OrchestrationManager()
    orch.pending_order2_plan = plan

    # Mock agent and room
    mock_agent = MagicMock()
    mock_agent.llm = MagicMock()

    # Mock streaming response from LLM
    class MockChunk:
        def __init__(self, text):
            self.choices = [MagicMock(delta=MagicMock(content=text))]

    async def mock_chat(chat_ctx):
        yield MockChunk("Jaise ki self-driving car ya voice assistant.")

    mock_agent.llm.chat = mock_chat
    mock_agent.say = AsyncMock()

    class MockParticipant:
        def __init__(self):
            self.published = []
        async def publish_data(self, data, topic="", reliable=True):
            self.published.append((topic, data))

    mock_room = MagicMock()
    mock_room.local_participant = MockParticipant()

    await agent._execute_order2_response(
        mock_agent,
        plan,
        history,
        mock_room,
        first_bot_content="AI ek technology hai...",
        orchestrator=orch,
        is_sathi=True,
    )

    # Verify Sathi spoke via agent.say
    mock_agent.say.assert_awaited_once()
    spoken_text = mock_agent.say.call_args[0][0]
    assert "self-driving car" in spoken_text

    # Verify Sathi turn added to shared history
    bot_turns = [t for t in history.turns if t.display_name == "Roxstar AI Sathi" or t.speaker == "sathi"]
    assert len(bot_turns) == 1
    assert "self-driving car" in bot_turns[0].text

    # Verify topic="chat" was NOT published (voice flow by default)
    published_topics = [t[0] for t in mock_room.local_participant.published]
    assert "chat" not in published_topics
    assert "room_context" in published_topics


@pytest.mark.anyio
async def test_order2_suppressed_if_interrupted():
    """Proves: If interrupted before execution, order 2 does not generate or speak."""
    history = RoomHistoryManager()
    plan = parse_ordered_plan("AI Dost, tum answer karo. AI Sathi, baad mein ek example dena.")
    orch = OrchestrationManager()
    orch.is_interrupted = True  # Simulated interruption

    mock_agent = MagicMock()
    mock_agent.llm = MagicMock()
    mock_agent.say = AsyncMock()
    mock_room = MagicMock()

    await agent._execute_order2_response(
        mock_agent,
        plan,
        history,
        mock_room,
        orchestrator=orch,
    )

    # say() must NOT be called
    mock_agent.say.assert_not_awaited()


@pytest.mark.anyio
async def test_order2_voice_does_not_publish_to_chat():
    """Proves: For microphone/voice interaction, order 2 response is NOT published to chat UI."""
    history = RoomHistoryManager()
    plan = parse_ordered_plan("AI Dost, tum answer karo. AI Sathi, simple example dena.")
    orch = OrchestrationManager()
    orch.pending_order2_plan = plan

    mock_agent = MagicMock()
    mock_agent.llm = MagicMock()

    class MockChunk:
        def __init__(self, text):
            self.choices = [MagicMock(delta=MagicMock(content=text))]

    async def mock_chat(chat_ctx):
        yield MockChunk("Jaise Alexa ya Siri ek example hai.")

    mock_agent.llm.chat = mock_chat
    mock_agent.say = AsyncMock()

    class MockParticipant:
        def __init__(self):
            self.published = []
        async def publish_data(self, data, topic="", reliable=True):
            self.published.append((topic, data))

    mock_room = MagicMock()
    mock_room.local_participant = MockParticipant()

    # Call with is_text_chat=False (default for voice)
    await agent._execute_order2_response(
        mock_agent,
        plan,
        history,
        mock_room,
        first_bot_content="AI machines ko train karta hai.",
        orchestrator=orch,
        is_sathi=True,
        is_text_chat=False,
    )

    # 1. Spoken via agent.say
    mock_agent.say.assert_awaited_once()
    # 2. Saved in RoomHistory
    bot_turns = [t for t in history.turns if "Sathi" in t.display_name or t.speaker == "sathi"]
    assert len(bot_turns) == 1
    # 3. NOT published to topic="chat"
    published_topics = [t[0] for t in mock_room.local_participant.published]
    assert "chat" not in published_topics
    assert "room_context" in published_topics


@pytest.mark.anyio
async def test_order2_text_chat_publishes_to_chat():
    """Proves: Explicit text chat interaction DOES publish order 2 response to chat UI."""
    history = RoomHistoryManager()
    plan = parse_ordered_plan("AI Dost, tum answer karo. AI Sathi, simple example dena.")
    orch = OrchestrationManager()
    orch.pending_order2_plan = plan

    mock_agent = MagicMock()
    mock_agent.llm = MagicMock()

    class MockChunk:
        def __init__(self, text):
            self.choices = [MagicMock(delta=MagicMock(content=text))]

    async def mock_chat(chat_ctx):
        yield MockChunk("Jaise Alexa ya Siri ek example hai.")

    mock_agent.llm.chat = mock_chat
    mock_agent.say = AsyncMock()

    class MockParticipant:
        def __init__(self):
            self.published = []
        async def publish_data(self, data, topic="", reliable=True):
            self.published.append((topic, data))

    mock_room = MagicMock()
    mock_room.local_participant = MockParticipant()

    # Call with is_text_chat=True (for explicit text chat)
    await agent._execute_order2_response(
        mock_agent,
        plan,
        history,
        mock_room,
        first_bot_content="AI machines ko train karta hai.",
        orchestrator=orch,
        is_sathi=True,
        is_text_chat=True,
    )

    # 1. Spoken via agent.say
    mock_agent.say.assert_awaited_once()
    # 2. Saved in RoomHistory
    bot_turns = [t for t in history.turns if "Sathi" in t.display_name or t.speaker == "sathi"]
    assert len(bot_turns) == 1
    # 3. Published to topic="chat"
    published_topics = [t[0] for t in mock_room.local_participant.published]
    assert "chat" in published_topics
    assert "room_context" in published_topics
