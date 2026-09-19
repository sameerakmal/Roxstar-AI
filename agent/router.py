import re
import logging

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
