import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from router import route_turn

def test_explicit_dost_addressing_english():
    assert route_turn("Dost, explain AI") == "dost"

def test_explicit_sathi_addressing_english():
    assert route_turn("Sathi, explain AI") == "sathi"
    assert route_turn("Saathi, explain AI") == "sathi"

def test_explicit_dost_addressing_hinglish():
    assert route_turn("Dost mujhe simple mein samjhao") == "dost"

def test_explicit_sathi_addressing_hinglish():
    assert route_turn("Sathi iska example do") == "sathi"
    assert route_turn("Saathi iska example do") == "sathi"

def test_unaddressed_english_question_defaults_to_dost():
    assert route_turn("Can you explain cloud computing?") == "dost"

def test_explicit_sathi_addressing_english_complex():
    assert route_turn("Sathi, can you explain cloud computing?") == "sathi"

def test_case_insensitive_addressing():
    assert route_turn("dost, AI kya hota hai?") == "dost"
    assert route_turn("SATHI, AI kya hota hai?") == "sathi"
    assert route_turn("hey DOST kaise ho?") == "dost"

def test_both_names_explicitly_mentioned():
    assert route_turn("Dost aur Sathi, dono batao") == "both"
    assert route_turn("You both explain this") == "both"
    assert route_turn("Dost and Sathi please give details") == "both"
    assert route_turn("दोस्त और साथी, दोनों बताओ") == "both"

def test_no_address_default_behavior():
    assert route_turn("AI kya hota hai?") == "dost"
    assert route_turn("What is machine learning?") == "dost"

def test_unaddressed_never_returns_both():
    result = route_turn("Explain quantum physics in detail")
    assert result != "both"
    assert result == "dost"

def test_generic_dost_word_usage_not_explicit_addressing():
    # "mere ek dost ne..." uses dost as a generic noun phrase, so it defaults to dost
    assert route_turn("Mere ek dost ne AI ke baare mein bataya") == "dost"
    assert route_turn("Mera dost kal aaya tha") == "dost"
    assert route_turn("मेरा एक दोस्त है") == "dost"

def test_devanagari_explicit_sathi_addressing():
    assert route_turn(" साथी AI क्या हहोता है?") == "sathi"
    assert route_turn("साथी, AI क्या होता है?") == "sathi"
    assert route_turn("हे साथी, समझाओ") == "sathi"
    assert route_turn("साथी मुझे बताओ") == "sathi"

def test_devanagari_explicit_dost_addressing():
    assert route_turn("दोस्त, AI क्या होता है?") == "dost"
    assert route_turn("हे दोस्त, समझाओ") == "dost"
    assert route_turn("दोस्त मुझे बताओ") == "dost"

def test_empty_or_whitespace_input():
    assert route_turn("") == "dost"
    assert route_turn("   ") == "dost"
