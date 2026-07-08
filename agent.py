"""
Baseline competition agent for PTCGABC.

The Kaggle simulation track expects this file to expose exactly:

    def agent(obs_dict: dict) -> list[int]

This implementation is intentionally dependency-light and works directly with
the raw observation dictionary. It is a safe heuristic baseline, not a trained
policy.
"""

from __future__ import annotations


# OptionType integer values from the cabt API.
OPTION_NUMBER = 0
OPTION_YES = 1
OPTION_NO = 2
OPTION_CARD = 3
OPTION_TOOL_CARD = 4
OPTION_ENERGY_CARD = 5
OPTION_ENERGY = 6
OPTION_PLAY = 7
OPTION_ATTACH = 8
OPTION_EVOLVE = 9
OPTION_ABILITY = 10
OPTION_DISCARD = 11
OPTION_RETREAT = 12
OPTION_ATTACK = 13
OPTION_END = 14
OPTION_SKILL = 15
OPTION_SPECIAL_CONDITION = 16


MAIN_ACTION_PRIORITY = {
    OPTION_ATTACK: 0,
    OPTION_EVOLVE: 1,
    OPTION_PLAY: 2,
    OPTION_ATTACH: 3,
    OPTION_ABILITY: 4,
    OPTION_SKILL: 5,
    OPTION_RETREAT: 6,
    OPTION_END: 20,
}

GENERIC_ACTION_PRIORITY = {
    OPTION_YES: 0,
    OPTION_CARD: 1,
    OPTION_ENERGY_CARD: 2,
    OPTION_TOOL_CARD: 3,
    OPTION_ENERGY: 4,
    OPTION_SPECIAL_CONDITION: 5,
    OPTION_NUMBER: 6,
    OPTION_NO: 10,
    OPTION_DISCARD: 12,
    OPTION_END: 20,
}


def agent(obs_dict: dict) -> list[int]:
    """
    Return selected option indices for the current cabt observation.

    The engine only presents legal options. For multi-select prompts, the agent
    chooses the best-scored option first and then fills the remaining required
    slots with the next best distinct options.
    """
    select = (obs_dict or {}).get("select") or {}
    options = select.get("option") or []
    max_count = int(select.get("maxCount", 0) or 0)
    if not options or max_count <= 0:
        return []

    ranked = sorted(range(len(options)), key=lambda i: (_score_option(options[i]), i))
    return ranked[: min(max_count, len(ranked))]


def _score_option(option: dict) -> int:
    option_type = option.get("type")

    # Main-turn options are where avoiding premature END matters most.
    if option_type in MAIN_ACTION_PRIORITY:
        score = MAIN_ACTION_PRIORITY[option_type]
    else:
        score = GENERIC_ACTION_PRIORITY.get(option_type, 8)

    # Prefer low numeric counts when the engine asks for a number. This is
    # conservative for damage/discard prompts and deterministic for testing.
    if option_type == OPTION_NUMBER:
        number = option.get("number", option.get("count", 0)) or 0
        score = score * 100 + int(number)

    return score
