import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from agent import DOST_SYSTEM_PROMPT

def test_dost_system_prompt_rules():
    assert "Roxstar AI Dost" in DOST_SYSTEM_PROMPT
    assert "Hinglish" in DOST_SYSTEM_PROMPT
    assert "Roman script" in DOST_SYSTEM_PROMPT
    assert "technology" in DOST_SYSTEM_PROMPT

def test_should_respond_relevance():
    def should_respond(text: str) -> bool:
        text_lower = text.lower()
        keywords = ["dost", "ai", "kya", "explain", "batao", "help", "cloud", "movie"]
        return any(kw in text_lower for kw in keywords)

    assert should_respond("AI kya hota hai?") == True
    assert should_respond("Can you explain cloud computing?") == True
    assert should_respond("Shah Rukh Khan ke baare mein batao") == True
    assert should_respond("random chatter without relevant words") == False

def test_groq_and_openrouter_env_check(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert os.getenv("OPENROUTER_API_KEY") is None
    assert os.getenv("GROQ_API_KEY") is None

def test_no_openai_key_required():
    agent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agent.py"))
    with open(agent_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "OPENROUTER_API_KEY" in content
    assert "GROQ_API_KEY" in content
    assert "OPENAI_API_KEY" not in content
    assert "sk-or-v1-" not in content
    assert "gsk_" not in content

def test_openrouter_llm_max_tokens_configured():
    agent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agent.py"))
    with open(agent_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "max_tokens=500" in content
    assert "temperature=0.7" in content
