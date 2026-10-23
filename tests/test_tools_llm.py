import datetime

import anthropic
import httpx
import pytest

from bmo.llm import (REFUSAL_REPLY, TRUNCATED_REPLY, MissingAPIKeyError, describe_api_error,
                     stop_reason_reply)
from bmo.tools import TOOLS, TOOLS_WITHOUT_CAMERA, current_time_text, format_search_results


def test_time_includes_the_date_and_weekday():
    now = datetime.datetime(2026, 10, 21, 15, 5, tzinfo=datetime.timezone.utc)
    assert current_time_text(now) == "It is 3:05 PM on Wednesday, October 21, 2026 (UTC)."


def test_time_without_a_zone_name():
    now = datetime.datetime(2026, 1, 2, 9, 0)
    assert current_time_text(now) == "It is 9:00 AM on Friday, January 2, 2026."


def test_search_results_keep_the_top_three_and_cap_snippets():
    results = [{"title": f"T{i}", "body": "word " * 200, "date": "2026-10-21T08:00:00"} for i in range(5)]
    text = format_search_results("news", results)
    lines = text.split("\n")
    assert lines[0] == "SEARCH RESULTS for 'news':"
    assert len(lines) == 4
    assert lines[1].startswith("1. T0 (2026-10-21): ")
    assert all(len(line) < 300 for line in lines[1:])


def test_search_results_accept_text_results_and_nothing():
    assert "2. B: second" in format_search_results("q", [{"title": "A", "snippet": "first"},
                                                         {"title": "B", "body": "second"}])
    assert format_search_results("q", []) == ""


def test_camera_tool_can_be_left_out():
    assert {t["name"] for t in TOOLS} - {t["name"] for t in TOOLS_WITHOUT_CAMERA} == {"capture_image"}


def _status_error(cls, status):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


@pytest.mark.parametrize("error,expected", [
    (MissingAPIKeyError("no key"), "No API key"),
    (_status_error(anthropic.AuthenticationError, 401), "rejected the API key"),
    (_status_error(anthropic.NotFoundError, 404), "Unknown model claude-x"),
    (_status_error(anthropic.BadRequestError, 400), "rejected the request"),
    (_status_error(anthropic.RateLimitError, 429), "rate limiting"),
    (_status_error(anthropic.InternalServerError, 500), "having trouble"),
    (anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), "check the network"),
    (RuntimeError("other"), "see agent.log"),
])
def test_describe_api_error(error, expected):
    message = describe_api_error(error, "claude-x")
    assert expected in message
    assert len(message) <= 64  # fits the 800 px status line


def test_stop_reason_reply():
    assert stop_reason_reply("refusal", True) == REFUSAL_REPLY
    assert stop_reason_reply("max_tokens", False) == TRUNCATED_REPLY
    assert stop_reason_reply("max_tokens", True) is None
    assert stop_reason_reply("end_turn", False) is None
    assert stop_reason_reply("tool_use", False) is None
