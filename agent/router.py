import re
import time
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("roxstar-ai-router")

# Regex for generic noun phrase usage in Hindi/Hinglish (Roman & Devanagari)
# e.g. "mere ek dost ne", "mera dost", "apne sathi", "मेरा एक दोस्त", "मेरा साथी"
GENERIC_DOST_PATTERN = re.compile(
    r'(?:mera|mere|meri|apna|apne|apni|ek|koi|humara|humare|humari|मेरा|मेरे|मेरी|अपना|अपने|अपनी|एक|कोई|हमारा|हमारे|हमारी)\s+(?:ek\s+|एक\s+)?(?:dost|dostt|दोस्त)\b',
    re.IGNORECASE
)

GENERIC_SATHI_PATTERN = re.compile(
    r'(?:mera|mere|meri|apna|apne|apni|ek|koi|humara|humare|humari|मेरा|मेरे|मेरी|अपना|अपने|अपनी|एक|कोई|हमारा|हमारे|हमारी)\s+(?:ek\s+|एक\s+)?(?:sathi|saathi|sati|saati|साथी|साती|सती)\b',
    re.IGNORECASE
)

# Phase 7: Referential mention patterns — bot name used as a subject/possessive reference, not an address
# e.g., "Dost ne jo bola", "Sathi ka example", "Dost ki baat", "Sathi ke baare mein"
REFERENTIAL_DOST_PATTERN = re.compile(
    r'(?:^|\s|[.,!?|:\-"\'])(?:dost|dostt|दोस्त)\s+(?:ne|ka|ki|ke|ने|का|की|के)(?:$|\s|[.,!?|:\-"\'])',
    re.IGNORECASE
)

REFERENTIAL_SATHI_PATTERN = re.compile(
    r'(?:^|\s|[.,!?|:\-"\'])(?:sathi|saathi|sati|saati|साथी|साती|सती)\s+(?:ne|ka|ki|ke|ने|का|की|के)(?:$|\s|[.,!?|:\-"\'])',
    re.IGNORECASE
)

# Both-bot trigger keywords/phrases (Roman & Devanagari)
BOTH_TRIGGERS = [
    r'(?:dono|दोनों|दोनो)\b',
    r'\bboth\b',
    r'(?:dost|dostt|दोस्त)\s+(?:aur|and|और)\s+(?:sathi|saathi|sati|saati|साथी|साती|सती)',
    r'(?:sathi|saathi|sati|saati|साथी|साती|सती)\s+(?:aur|and|और)\s+(?:dost|dostt|दोस्त)',
    r'you\s+both',
    r'both\s+of\s+you',
]

BOTH_PATTERN = re.compile('|'.join(BOTH_TRIGGERS), re.IGNORECASE)

# Phase 7: Order keyword patterns for multi-bot sequencing
FIRST_ORDER_KEYWORDS = re.compile(
    r'\b(?:pehle|pahle|first|पहले)\b',
    re.IGNORECASE
)

SECOND_ORDER_KEYWORDS = re.compile(
    r'(?:\bbaad\s+(?:mein|me)\b|\bafterwards?\b|\bthen\b|\bphir\b|\buske\s+baad\b|\biske\s+baad\b|बाद\s+में|फिर|उसके\s+बाद|इसके\s+बाद)',
    re.IGNORECASE
)

# Bot name patterns for position detection in multi-bot plans
DOST_NAME_PATTERN = re.compile(r'\b(?:dost|dostt|दोस्त)\b', re.IGNORECASE)
SATHI_NAME_PATTERN = re.compile(r'\b(?:sathi|saathi|sati|saati|साथी|साती|सती)\b', re.IGNORECASE)


# ── Phase 7: Multi-bot ordered plan data structures ──────────────────────────

@dataclass
class MultiBotTarget:
    """Represents a single bot's role in an ordered multi-bot plan."""
    bot: str          # "dost" or "sathi"
    order: int        # 1 or 2
    instruction: str  # Extracted instruction clause for this bot


@dataclass
class MultiBotPlan:
    """Ordered multi-bot plan extracted from user text."""
    turn_id: str
    targets: List[MultiBotTarget]
    timestamp: float
    original_text: str


def _generate_plan_turn_id(text: str) -> str:
    """
    Generate a deterministic turn ID from user text so both workers produce the same ID.
    Uses a 60-second time bucket so workers processing the same STT text independently
    will generate an identical turn_id within the same window.
    """
    text_hash = hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()[:12]
    time_bucket = int(time.time()) // 60
    return f"plan_{text_hash}_{time_bucket}"


# ── Addressing detection ─────────────────────────────────────────────────────

def is_explicit_address(text: str, name: str) -> bool:
    """
    Determines if 'text' explicitly addresses 'name' ('dost' or 'sathi')
    as a participant in Roman or Devanagari script.
    """
    text_lower = text.lower().strip()
    
    name_pattern = r'(sathi|saathi|sati|saati|साथी|साती|सती)' if name == 'sathi' else r'(dost|dostt|दोस्त)'
    
    # Check if name pattern is present in the text (using boundary or whitespace/punctuation check)
    boundary_pattern = r'(?:^|\s|[.,!?|:\-"\'])' + name_pattern + r'(?:$|\s|[.,!?|:\-"\'])'
    if not re.search(boundary_pattern, text_lower):
        return False

    # Phase 7: Check for referential mentions (e.g., "Dost ne jo bola", "Sathi ka example")
    # If ALL occurrences of the name are referential, it is NOT an address
    referential_pattern = REFERENTIAL_DOST_PATTERN if name != 'sathi' else REFERENTIAL_SATHI_PATTERN
    text_without_refs = referential_pattern.sub(' ', text_lower).strip()
    if not re.search(boundary_pattern, text_without_refs):
        logger.debug(f"[Router] Name '{name}' is referential-only in: \"{text}\"")
        return False
        
    # Strong explicit addressing cues
    strong_cues = [
        r'^\s*' + name_pattern + r'(?:$|\s|[.,!?|:\-"\'])',    # Starts with name (e.g. "साथी", "Sathi, explain")
        r'(?:hey|hi|hello|हे|हाय|हेलो|अरे|सुनो)\s+' + name_pattern,
        r'(?:roxstar|roxstar\s+ai|रॉक्सस्टार|रॉकस्टार)\s+' + name_pattern,
        name_pattern + r'\s*[,!?:;]',
        name_pattern + r'\s+(?:mujhe|batao|bataiye|samjhao|samjaiye|kya|can|explain|iska|मुझे|बताओ|बताइए|समझाओ|समझाइए|क्या|का|की|के)',
    ]
    
    for cue in strong_cues:
        if re.search(cue, text_lower):
            return True
            
    # Check if used as a generic noun phrase (e.g. "mere ek dost ne", "मेरा एक दोस्त")
    generic_pattern = GENERIC_DOST_PATTERN if name == "dost" else GENERIC_SATHI_PATTERN
    if generic_pattern.search(text_lower):
        return False
        
    # If name appears standalone without generic qualifiers, treat as explicit address
    return True


# ── Turn routing ─────────────────────────────────────────────────────────────

def route_turn(text: str) -> str:
    """
    Determines which agent should respond to the user utterance/message.
    
    Returns:
        "dost": Route turn to Roxstar AI Dost (default)
        "sathi": Route turn to Roxstar AI Sathi
        "both": Route turn to both agents sequentially
    """
    if not text or not text.strip():
        logger.info("[Router] Empty text turn. Defaulting to: roxstar-ai-dost")
        return "dost"

    text_clean = text.strip()

    # Check for explicit both-bot triggers (e.g. "dono", "दोनों", "you both", "dost aur sathi")
    has_both_trigger = bool(BOTH_PATTERN.search(text_clean))
    
    dost_addressed = is_explicit_address(text_clean, "dost")
    sathi_addressed = is_explicit_address(text_clean, "sathi")

    if has_both_trigger or (dost_addressed and sathi_addressed):
        logger.info(f'[Router] Transcript: "{text_clean}" -> Selected agent: BOTH (Sequential)')
        return "both"

    if sathi_addressed and not dost_addressed:
        logger.info(f'[Router] Transcript: "{text_clean}" -> Selected agent: roxstar-ai-sathi')
        return "sathi"

    if dost_addressed and not sathi_addressed:
        logger.info(f'[Router] Transcript: "{text_clean}" -> Selected agent: roxstar-ai-dost')
        return "dost"

    # Default routing: unaddressed questions or generic word usage go to Dost
    logger.info(f'[Router] Transcript: "{text_clean}" -> No explicit address match. Defaulting to: roxstar-ai-dost')
    return "dost"


# ── Phase 7: Ordered multi-bot plan parsing ──────────────────────────────────

def parse_ordered_plan(text: str) -> Optional[MultiBotPlan]:
    """
    Parse user text into an ordered multi-bot plan.
    Called only when route_turn() returns 'both'.
    
    Determines which bot speaks first by:
    1. Checking explicit order keywords (pehle/first vs baad mein/afterwards)
    2. Falling back to positional order (which bot name appears first)
    
    Extracts per-bot instruction clauses from the text.
    """
    text_clean = text.strip()
    if not text_clean:
        return None

    # Split into clauses by sentence boundaries (period, Devanagari danda)
    clauses = re.split(r'[.।]+', text_clean)
    clauses = [c.strip() for c in clauses if c.strip()]

    # If still one clause, try splitting by comma with space
    if len(clauses) == 1:
        clauses = re.split(r',\s+', text_clean)
        clauses = [c.strip() for c in clauses if c.strip()]

    # Attribute clauses to bots
    dost_clauses: List[str] = []
    sathi_clauses: List[str] = []

    for clause in clauses:
        has_dost = bool(DOST_NAME_PATTERN.search(clause))
        has_sathi = bool(SATHI_NAME_PATTERN.search(clause))

        if has_dost and not has_sathi:
            dost_clauses.append(clause)
        elif has_sathi and not has_dost:
            sathi_clauses.append(clause)
        elif has_dost and has_sathi:
            # Both names in clause: check if one is referential
            dost_is_ref = bool(REFERENTIAL_DOST_PATTERN.search(clause))
            sathi_is_ref = bool(REFERENTIAL_SATHI_PATTERN.search(clause))

            if dost_is_ref and not sathi_is_ref:
                sathi_clauses.append(clause)
            elif sathi_is_ref and not dost_is_ref:
                dost_clauses.append(clause)
            # else: neither or both referential → unattributed (falls back to full text)

    # Build per-bot instruction text
    dost_instruction = " ".join(dost_clauses) if dost_clauses else text_clean
    sathi_instruction = " ".join(sathi_clauses) if sathi_clauses else text_clean

    # Determine order from explicit keywords within each bot's clauses
    dost_text_combined = " ".join(dost_clauses).lower() if dost_clauses else ""
    sathi_text_combined = " ".join(sathi_clauses).lower() if sathi_clauses else ""

    dost_has_first = bool(FIRST_ORDER_KEYWORDS.search(dost_text_combined)) if dost_text_combined else False
    dost_has_second = bool(SECOND_ORDER_KEYWORDS.search(dost_text_combined)) if dost_text_combined else False
    sathi_has_first = bool(FIRST_ORDER_KEYWORDS.search(sathi_text_combined)) if sathi_text_combined else False
    sathi_has_second = bool(SECOND_ORDER_KEYWORDS.search(sathi_text_combined)) if sathi_text_combined else False

    # Determine order
    if sathi_has_first or dost_has_second:
        dost_order, sathi_order = 2, 1
    elif dost_has_first or sathi_has_second:
        dost_order, sathi_order = 1, 2
    else:
        # No explicit order keywords: use positional order in original text
        dost_match = DOST_NAME_PATTERN.search(text_clean)
        sathi_match = SATHI_NAME_PATTERN.search(text_clean)

        dost_pos = dost_match.start() if dost_match else len(text_clean)
        sathi_pos = sathi_match.start() if sathi_match else len(text_clean)

        if dost_pos <= sathi_pos:
            dost_order, sathi_order = 1, 2
        else:
            dost_order, sathi_order = 2, 1

    turn_id = _generate_plan_turn_id(text_clean)

    plan = MultiBotPlan(
        turn_id=turn_id,
        targets=[
            MultiBotTarget(bot="dost", order=dost_order, instruction=dost_instruction),
            MultiBotTarget(bot="sathi", order=sathi_order, instruction=sathi_instruction),
        ],
        timestamp=time.time(),
        original_text=text_clean,
    )

    logger.info(
        f'[Router] Ordered plan: turn_id={plan.turn_id} '
        f'dost_order={dost_order} sathi_order={sathi_order}'
    )
    return plan
