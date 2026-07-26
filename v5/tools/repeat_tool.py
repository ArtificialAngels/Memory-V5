"""V5.2: Anti-repeat tools for MCP exposure."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger("ikaros.v5.tools.repeat_tool")


def v5_anti_repeat_record(character: str, response_text: str) -> str:
    """Record a response into the anti-repetition corpus.

    Args:
        character: Character/role name
        response_text: The AI response to analyze

    Returns:
        JSON: {"recorded": <int>}
    """
    from v5 import anti_repeat
    n = anti_repeat.record_response(character, response_text)
    return json.dumps({"recorded": n})


def v5_anti_repeat_check(character: str, candidate_text: str) -> str:
    """Check if a candidate response has high repetition risk.

    Args:
        character: Character/role name
        candidate_text: Text to check

    Returns:
        JSON result with score, is_repetitive, top_ngrams
    """
    from v5 import anti_repeat
    result = anti_repeat.check_repetition(character, candidate_text)
    return json.dumps(result, ensure_ascii=False)


def v5_anti_repeat_penalty(character: str, candidate_text: str) -> str:
    """Get a penalty hint if repetition risk is high.

    Args:
        character: Character/role name
        candidate_text: Text to check

    Returns:
        Penalty hint string (empty if clean)
    """
    from v5 import anti_repeat
    return anti_repeat.get_penalty_hint(character, candidate_text)


def v5_anti_repeat_clear(character: str = "") -> str:
    """Clear anti-repeat corpus.

    Args:
        character: Character/role (empty = clear all)

    Returns:
        JSON: {"deleted": <int>}
    """
    from v5 import anti_repeat
    n = anti_repeat.clear(character)
    return json.dumps({"deleted": n})


def v5_anti_repeat_stats(character: str = "") -> str:
    """Get anti-repeat corpus statistics.

    Returns:
        JSON stats
    """
    from v5 import anti_repeat
    return json.dumps(anti_repeat.stats(character))
