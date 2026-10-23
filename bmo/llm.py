"""Claude API helpers: status-line text for API errors, and what to say when a
response ends without a usable reply."""

import anthropic


class MissingAPIKeyError(RuntimeError):
    pass


def describe_api_error(e, model):
    """Short status-line text (the 800 px status bar) for an exception from a
    Claude request. Distinguishes a bad key, a bad model or parameter, rate
    limiting, server trouble and a network failure, so the face says what to
    fix instead of blaming the network for everything. Details go to agent.log."""
    if isinstance(e, MissingAPIKeyError):
        return "No API key: set ANTHROPIC_API_KEY in .env"
    if isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return "Claude rejected the API key: check .env"
    if isinstance(e, anthropic.NotFoundError):
        return f"Unknown model {model}: check text_model"
    if isinstance(e, (anthropic.BadRequestError, anthropic.UnprocessableEntityError)):
        return "Claude rejected the request: check model, temperature, effort"
    if isinstance(e, anthropic.RateLimitError):
        return "Claude is rate limiting: try again soon"
    if isinstance(e, anthropic.APIStatusError) and e.status_code >= 500:
        return "Claude is having trouble: try again later"
    if isinstance(e, anthropic.APIConnectionError):  # includes APITimeoutError
        return "Can't reach Claude: check the network"
    return "Can't reach Claude: see agent.log"


# Spoken when a response stops without a reply the user can use. "refusal":
# the model declined (Claude Haiku 5.5 has no server-side fallback model);
# "max_tokens": the cap ran out, usually while thinking, before any text.
REFUSAL_REPLY = "Sorry, I can't help with that one."
TRUNCATED_REPLY = "Hmm, that one was too much for me. Can you ask a shorter way?"


def stop_reason_reply(stop_reason, produced_text):
    """What to add to a reply after the final message of a round, or None.
    A refusal always gets the apology, since a mid-reply refusal cuts the text
    off; a max_tokens stop only when nothing was said."""
    if stop_reason == "refusal":
        return REFUSAL_REPLY
    if stop_reason == "max_tokens" and not produced_text:
        return TRUNCATED_REPLY
    return None
