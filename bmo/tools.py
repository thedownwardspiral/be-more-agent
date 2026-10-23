"""Tool schemas offered to Claude, and the text the tools return."""

# How many web results search_web returns, and how much of each snippet
SEARCH_MAX_RESULTS = 3
SEARCH_SNIPPET_MAX = 250

TOOLS = [
    {
        "name": "get_time",
        "description": "Get the current local date, day of the week and time.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}
    },
    {
        "name": "search_web",
        "description": "Search the web for news or current information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
            "additionalProperties": False
        }
    },
    {
        "name": "capture_image",
        "description": "Take a photo with the onboard camera so you can see the surroundings.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}
    }
]
TOOL_NAMES = [t["name"] for t in TOOLS]
TOOLS_WITHOUT_CAMERA = [t for t in TOOLS if t["name"] != "capture_image"]


def current_time_text(now):
    """get_time's result for a timezone-aware datetime: "It is 3:05 PM on
    Wednesday, October 21, 2026 (PDT)." The date matters as much as the time:
    without it Claude has to guess "what day is it?"."""
    clock = now.strftime("%I:%M %p").lstrip("0")
    date = f"{now.strftime('%A, %B')} {now.day}, {now.year}"
    zone = now.tzname()
    return f"It is {clock} on {date}" + (f" ({zone})." if zone else ".")


def _cap(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def format_search_results(query, results):
    """search_web's result text for a list of ddgs result dicts (news items
    carry "body", text results "body" or "snippet"). Empty string for no
    results."""
    lines = []
    for r in results[:SEARCH_MAX_RESULTS]:
        title = _cap(r.get("title") or "No title", 120)
        body = _cap(r.get("body") or r.get("snippet") or "", SEARCH_SNIPPET_MAX)
        date = r.get("date")
        dated = f" ({str(date)[:10]})" if date else ""
        lines.append(f"{len(lines) + 1}. {title}{dated}: {body}")
    if not lines:
        return ""
    return f"SEARCH RESULTS for '{query}':\n" + "\n".join(lines)
