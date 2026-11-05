from voice_agent.script import Background, Effect, Pause, Speech, parse_duration, parse_script, strip_tags


def test_parse_all_tags():
    ev = parse_script("[bg:rain] Hello [sfx:thunder 3s] [pause:250ms] [voice:male] [lang:te] నమస్కారం [bg:off]")
    assert ev == [
        Background("rain"),
        Speech("Hello"),
        Effect("thunder", 3.0),
        Pause(0.25),
        Speech("నమస్కారం", voice="male", language="te"),
        Background(None),
    ]


def test_telugu_effect_prompt_and_default_pause():
    ev = parse_script("[sfx:గుడి గంట][pause]")
    assert ev == [Effect("గుడి గంట", None), Pause(0.5)]


def test_durations():
    assert parse_duration("1.5s") == 1.5
    assert parse_duration("800ms") == 0.8
    assert parse_duration("2") == 2.0
    assert parse_duration("soon", 0.3) == 0.3


def test_strip_tags():
    assert strip_tags("[sfx:chime] Hi   there [pause:1s]!") == "Hi there !"
