"""
Roxstar AI Voice Room Assistant - Shared Room Context Manager (Phase 4)

Provides lightweight, in-memory room history management for LiveKit rooms.
Handles canonical turn ordering, unique turn_id idempotency, speaker identity formatting,
sliding window bounds, and DataChannel serialization/deserialization.
"""

import time
import uuid
import json
import re
import logging
from dataclasses import dataclass, asdict
from typing import List, Set, Optional, Dict, Any
from livekit.agents import llm

logger = logging.getLogger("roxstar-ai-room-context")

SPEAKER_HUMAN = "human"
SPEAKER_DOST = "dost"
SPEAKER_SATHI = "sathi"

DISPLAY_NAME_HUMAN = "Human"
DISPLAY_NAME_DOST = "Roxstar AI Dost"
DISPLAY_NAME_SATHI = "Roxstar AI Sathi"

SPEAKER_LABEL_PATTERN = re.compile(
    r'^\s*\[?(?:Roxstar\s+AI\s+)?(?:Dost|Sathi|Human)\]?\s*:\s*',
    re.IGNORECASE
)

def strip_speaker_labels(text: str) -> str:
    """
    Strips internal speaker metadata labels from text, preventing
    labels from leaking into raw stored history or speech synthesis.
    Handles single, bracketed, unbracketed, and repeated labels.
    """
    if not text:
        return ""
    cleaned = text.strip()
    while True:
        match = SPEAKER_LABEL_PATTERN.match(cleaned)
        if not match:
            break
        cleaned = cleaned[match.end():].strip()
    return cleaned

@dataclass
class RoomTurn:
    turn_id: str
    speaker: str  # "human", "dost", or "sathi"
    display_name: str  # "Human", "Roxstar AI Dost", "Roxstar AI Sathi"
    text: str
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoomTurn":
        raw_text = strip_speaker_labels(str(data.get("text", "")))
        return cls(
            turn_id=str(data.get("turn_id", "")),
            speaker=str(data.get("speaker", SPEAKER_HUMAN)),
            display_name=str(data.get("display_name", DISPLAY_NAME_HUMAN)),
            text=raw_text,
            timestamp=float(data.get("timestamp", time.time())),
        )

    def get_speaker_prefix(self) -> str:
        if self.speaker == SPEAKER_DOST:
            return f"[{DISPLAY_NAME_DOST}]"
        elif self.speaker == SPEAKER_SATHI:
            return f"[{DISPLAY_NAME_SATHI}]"
        else:
            return f"[{DISPLAY_NAME_HUMAN}]"


class RoomHistoryManager:
    """
    In-memory room history manager for a LiveKit room.
    Maintains ordered turns, turn_id idempotency, sliding window,
    and formats ChatContext for LLMs with raw turn text.
    """
    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.turns: List[RoomTurn] = []
        self.seen_turn_ids: Set[str] = set()

    def add_turn(self, turn: RoomTurn) -> bool:
        """
        Adds a canonical turn to room history if turn_id is not already seen.
        Ensures turn text is stored as raw text without internal metadata labels.
        Returns True if turn was added, False if ignored as duplicate.
        """
        if not turn.turn_id:
            turn.turn_id = f"{turn.speaker}_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"

        if turn.turn_id in self.seen_turn_ids:
            logger.debug(f"[RoomHistory] Duplicate turn_id '{turn.turn_id}' ignored.")
            return False

        # Guarantee raw turn text without internal speaker prefixes
        turn.text = strip_speaker_labels(turn.text)

        self.seen_turn_ids.add(turn.turn_id)
        self.turns.append(turn)
        logger.info(f"[RoomHistory] Turn added: turn_id={turn.turn_id} speaker={turn.speaker} text='{turn.text[:40]}...'")

        # Maintain sliding window bound
        while len(self.turns) > self.max_turns:
            old_turn = self.turns.pop(0)
            logger.debug(f"[RoomHistory] Evicted oldest turn '{old_turn.turn_id}' due to max_turns limit ({self.max_turns}).")

        return True

    def create_human_turn(self, text: str, turn_id: Optional[str] = None) -> RoomTurn:
        """Helper to create a canonical human turn with a unique turn_id and raw text."""
        if not turn_id:
            turn_id = f"human_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
        clean_text = strip_speaker_labels(text)
        return RoomTurn(
            turn_id=turn_id,
            speaker=SPEAKER_HUMAN,
            display_name=DISPLAY_NAME_HUMAN,
            text=clean_text,
            timestamp=time.time(),
        )

    def create_bot_turn(self, speaker: str, text: str, turn_id: Optional[str] = None) -> RoomTurn:
        """Helper to create a canonical bot turn with a unique turn_id and raw text."""
        if not turn_id:
            turn_id = f"{speaker}_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
        display_name = DISPLAY_NAME_SATHI if speaker == SPEAKER_SATHI else DISPLAY_NAME_DOST
        clean_text = strip_speaker_labels(text)
        return RoomTurn(
            turn_id=turn_id,
            speaker=speaker,
            display_name=display_name,
            text=clean_text,
            timestamp=time.time(),
        )

    def build_chat_context(self, system_prompt: str, is_sathi: bool = False) -> llm.ChatContext:
        """
        Constructs an LLM ChatContext containing the system prompt and ordered turns.
        Speaker identity prefixes ([Human], [Roxstar AI Dost], [Roxstar AI Sathi])
        are added strictly here, exactly once per turn.
        """
        chat_ctx = llm.ChatContext()
        chat_ctx.append(role="system", text=system_prompt)

        for turn in self.turns:
            prefix = turn.get_speaker_prefix()
            clean_text = strip_speaker_labels(turn.text)
            formatted_content = f"{prefix}: {clean_text}"

            if turn.speaker == SPEAKER_HUMAN:
                chat_ctx.append(role="user", text=formatted_content)
            else:
                # Bot turns (Dost or Sathi) are represented as assistant messages
                chat_ctx.append(role="assistant", text=formatted_content)

        return chat_ctx

    def serialize_turn_event(self, turn: RoomTurn) -> bytes:
        """Serializes a RoomTurn into a JSON DataChannel packet with raw text."""
        turn_copy = RoomTurn(
            turn_id=turn.turn_id,
            speaker=turn.speaker,
            display_name=turn.display_name,
            text=strip_speaker_labels(turn.text),
            timestamp=turn.timestamp,
        )
        payload = {
            "type": "room_turn",
            "turn": turn_copy.to_dict(),
        }
        return json.dumps(payload).encode("utf-8")

    @classmethod
    def deserialize_turn_event(cls, data_bytes: bytes) -> Optional[RoomTurn]:
        """Deserializes DataChannel bytes into a RoomTurn if valid room_turn type."""
        try:
            raw_str = data_bytes.decode("utf-8")
            data = json.loads(raw_str)
            if data.get("type") == "room_turn" and "turn" in data:
                return RoomTurn.from_dict(data["turn"])
        except Exception as e:
            logger.debug(f"[RoomHistory] Failed to deserialize DataChannel turn event: {e}")
        return None

    def clear(self):
        """Clears all turns and seen turn IDs."""
        self.turns.clear()
        self.seen_turn_ids.clear()

