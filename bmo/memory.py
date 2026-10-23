"""memory.json entries: tool-call records, validation, and rendering the saved
conversation as Anthropic messages."""

import re

# A saved reply's tool calls ("tool_calls" in memory.json), replayed to Claude
# as real tool_use / tool_result blocks so it can see that an earlier answer
# came from a tool. Kept small: inputs and results are capped, and neither
# photos nor web text are stored. Ids and names must fit the API's tool_use
# patterns, or every later request would be rejected.
TOOL_RECORD_INPUT_MAX = 200   # per string value in the tool input
TOOL_RECORD_RESULT_MAX = 600
TOOL_RECORD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Stored instead of a successful tool's output when that output is untrusted
# or large. Web text is third-party content: replaying it in every later turn,
# across restarts, would let a planted page keep steering the bot long after
# the search. The reply saved after the record already says what was found.
PHOTO_NOT_RETAINED = "photo captured (image not retained)"
SEARCH_NOT_RETAINED = "search results used for the reply below (web text not retained)"
RESULT_NOT_RETAINED = "(result not retained)"


def _cap_text(text, limit):
    return text if len(text) <= limit else text[:limit - 1] + "…"


def make_tool_record(tool_id, name, tool_input, result_content, is_error):
    """A compact, JSON-safe record of one executed tool call. tool_id is the
    API's own tool_use id (server-unique, so ids never collide in memory)."""
    if name == "search_web" and not is_error:
        result = SEARCH_NOT_RETAINED
    elif isinstance(result_content, str):
        result = result_content
    elif name == "capture_image" and not is_error:
        result = PHOTO_NOT_RETAINED
    else:
        result = RESULT_NOT_RETAINED
    if not isinstance(tool_input, dict):
        tool_input = {}
    return {
        "id": tool_id,
        "name": name,
        "input": {k: _cap_text(v, TOOL_RECORD_INPUT_MAX) if isinstance(v, str) else v
                  for k, v in tool_input.items()},
        "result": _cap_text(result, TOOL_RECORD_RESULT_MAX) or "(no output)",
        "error": bool(is_error),
    }


def is_valid_tool_record(r):
    return (isinstance(r, dict)
            and isinstance(r.get("id"), str) and bool(TOOL_RECORD_ID_RE.match(r["id"]))
            and isinstance(r.get("name"), str) and bool(TOOL_RECORD_ID_RE.match(r["name"]))
            and isinstance(r.get("input"), dict)
            and isinstance(r.get("result"), str)
            and isinstance(r.get("error"), bool))


def is_valid_memory(history):
    """True if history is a well-formed memory.json: a system entry first, then
    user/assistant entries only, each with non-empty string content. An
    assistant entry may carry an optional "tool_calls" list of tool records
    (see make_tool_record; ids unique across the file). Any entry may still
    carry a legacy "tools" list of strings from older builds, which
    upgrade_memory() drops."""
    if not isinstance(history, list) or not history:
        return False
    seen_ids = set()
    for i, m in enumerate(history):
        if not isinstance(m, dict):
            return False
        if m.get("role") not in ("system", "user", "assistant"):
            return False
        if (m["role"] == "system") != (i == 0):
            return False  # system first, and only first
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            return False
        if "tools" in m:
            tools = m["tools"]
            if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
                return False
        if "tool_calls" in m:
            calls = m["tool_calls"]
            if m["role"] != "assistant" or not isinstance(calls, list):
                return False
            for r in calls:
                if not is_valid_tool_record(r) or r["id"] in seen_ids:
                    return False
                seen_ids.add(r["id"])
    return True


def upgrade_memory(history):
    """Bring a valid memory.json from an older build up to date, in place.
    Returns (legacy tool labels dropped, search results scrubbed): older
    builds saved a "tools" list of labels, and kept web text in search
    records."""
    labels = scrubbed = 0
    for m in history:
        if m.pop("tools", None) is not None:
            labels += 1
        for r in m.get("tool_calls", ()):
            if r["name"] == "search_web" and not r["error"] and r["result"] != SEARCH_NOT_RETAINED:
                r["result"] = SEARCH_NOT_RETAINED
                scrubbed += 1
    return labels, scrubbed


def conversation_tail(entries, keep):
    """The last `keep` non-system entries of entries."""
    conv = [m for m in entries if m.get("role") != "system"]
    return conv[-keep:]


def history_messages(entries, max_messages):
    """Saved memory entries as Anthropic messages (system entries skipped).

    Uses the last `max_messages` entries (entries, not rendered messages: a
    reply with tool calls renders as three), starting at a user message. A
    reply with "tool_calls" is replayed as the calls that produced it: an
    assistant message of tool_use blocks, a user message of the matching
    tool_result blocks, then the reply text. No thinking blocks are stored or
    replayed, so there is nothing for the API to bind."""
    history = []
    for msg in conversation_tail(entries, max_messages):
        if not history and msg["role"] != "user":
            continue  # API conversations must start with a user message
        if msg["role"] == "assistant" and history[-1]["role"] == "assistant":
            continue  # memory never has two replies in a row; don't send one
        calls = msg.get("tool_calls") if msg["role"] == "assistant" else None
        if calls:
            history.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["input"]}
                for c in calls
            ]})
            results = []
            for c in calls:
                block = {"type": "tool_result", "tool_use_id": c["id"], "content": c["result"]}
                if c["error"]:
                    block["is_error"] = True
                results.append(block)
            history.append({"role": "user", "content": results})
        history.append({"role": msg["role"], "content": msg["content"]})
    return history
