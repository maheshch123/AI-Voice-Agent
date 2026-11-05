from voice_agent.language import ENGLISH, TELUGU, detect_language, segment_text


def test_detect_pure_languages():
    assert detect_language("నమస్కారం, ఎలా ఉన్నారు?") == TELUGU
    assert detect_language("Hello, how are you?") == ENGLISH
    assert detect_language("123 !!!") == ENGLISH


def test_segment_code_mixed_sentence():
    segs = segment_text("నమస్కారం! Welcome to Vani, మీ AI voice assistant.")
    assert [s.language for s in segs] == [TELUGU, ENGLISH, TELUGU, ENGLISH]
    assert segs[0].text.startswith("నమస్కారం")
    assert segs[1].text == "Welcome to Vani,"


def test_short_runs_are_merged():
    # A 2-letter English token inside Telugu shouldn't flip the voice.
    segs = segment_text("నేను TV చూస్తున్నాను")
    assert len(segs) == 1 and segs[0].language == TELUGU


def test_empty_and_whitespace():
    assert segment_text("") == []
    assert segment_text("   ") == []


def test_single_english_word_stays_in_telugu_voice():
    segs = segment_text("నేను office కి వెళ్తున్నాను. Then I will call you back.")
    assert [s.language for s in segs] == [TELUGU, ENGLISH]
    assert segs[0].text == "నేను office కి వెళ్తున్నాను."
