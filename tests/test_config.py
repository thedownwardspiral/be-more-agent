import json

from bmo.config import DEFAULT_CONFIG, load_config, validate_config


def test_valid_values_pass_through():
    cfg = {"whisper_threads": 2, "chat_memory": False, "effort": None, "input_device": "USB"}
    assert validate_config(cfg) == cfg


def test_unknown_and_invalid_keys_are_dropped(caplog):
    valid = validate_config({"whisper_thread": 2, "vad_threshold": 5, "camera_rotation": 45})
    assert valid == {}
    assert "unknown key" in caplog.text
    assert "invalid 'vad_threshold'" in caplog.text


def test_coercions():
    valid = validate_config({"chat_memory": "false", "whisper_threads": 3.0, "temperature": 1})
    assert valid == {"chat_memory": False, "whisper_threads": 3, "temperature": 1.0}


def test_bool_is_not_a_number():
    assert validate_config({"whisper_threads": True}) == {}


def test_load_config_merges_over_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"effort": "high", "bogus": 1}))
    cfg = load_config(path)
    assert cfg["effort"] == "high"
    assert "bogus" not in cfg
    assert cfg["text_model"] == DEFAULT_CONFIG["text_model"]


def test_unparseable_config_is_ignored(tmp_path, caplog):
    path = tmp_path / "config.json"
    path.write_text("{not json")
    assert load_config(path) == DEFAULT_CONFIG
    assert "IGNORING THE WHOLE FILE" in caplog.text


def test_missing_config_is_defaults(tmp_path):
    assert load_config(tmp_path / "absent.json") == DEFAULT_CONFIG


def test_transcripts_are_not_logged_by_default():
    assert DEFAULT_CONFIG["log_transcripts"] is False
