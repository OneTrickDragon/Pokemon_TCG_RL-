from __future__ import annotations
 
import json
 
import sim


def _decode_obs(raw: bytes | str | None) -> dict | None:
    """Decode a C-string JSON response into a Python dict."""
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not raw:
        return None
    return json.loads(raw)


 
def _check_result(obs_dict: dict | None, context: str) -> dict:
    """Raise on engine errors; return the obs_dict on success."""
    if obs_dict is None:
        raise RuntimeError(f"{context}: engine returned no data")
    error = obs_dict.get("error")
    if error:
        raise RuntimeError(f"{context} failed: {error}")
    return obs_dict


class StartData:
    """
    Python mirror of sim.StartData.
 
    Attributes
    ----------
    error : str | None
        None on success; error message on failure.
    player0 : int
        Index assigned to the first deck (almost always 0).
    player1 : int
        Index assigned to the second deck (almost always 1).
    """
    __slots__ = ("error", "player0", "player1")
 
    def __init__(self, error: str | None, player0: int, player1: int):
        self.error   = error
        self.player0 = player0
        self.player1 = player1
 
    def __repr__(self) -> str:
        if self.error:
            return f"StartData(error={self.error!r})"
        return f"StartData(player0={self.player0}, player1={self.player1})"
 
 
class StartData:
    """
    Python mirror of sim.StartData.
 
    Attributes
    ----------
    error : str | None
        None on success; error message on failure.
    player0 : int
        Index assigned to the first deck (almost always 0).
    player1 : int
        Index assigned to the second deck (almost always 1).
    """
    __slots__ = ("error", "player0", "player1")
 
    def __init__(self, error: str | None, player0: int, player1: int):
        self.error   = error
        self.player0 = player0
        self.player1 = player1
 
    def __repr__(self) -> str:
        if self.error:
            return f"StartData(error={self.error!r})"
        return f"StartData(player0={self.player0}, player1={self.player1})"
 
 
# ---------------------------------------------------------------------------
# High-level game functions  (public API — imported by ptcg_env.py)
# ---------------------------------------------------------------------------
 
def battle_start(
    deck0: list[int],
    deck1: list[int],
) -> tuple[dict | None, StartData]:
    """
    Start a new battle with two 60-card decks.
 
    Parameters
    ----------
    deck0 : list[int]
        Card IDs for player 0's deck (exactly 60).
    deck1 : list[int]
        Card IDs for player 1's deck (exactly 60).
 
    Returns
    -------
    (obs_dict, start_data)
        obs_dict is None when initialisation fails (check start_data.error).
 
    Raises
    ------
    ValueError
        If either deck does not contain exactly 60 card IDs.
    """
    if len(deck0) != 60:
        raise ValueError(f"deck0 must have 60 cards, got {len(deck0)}")
    if len(deck1) != 60:
        raise ValueError(f"deck1 must have 60 cards, got {len(deck1)}")
 
    battle = sim.Battle(deck0, deck1)
    raw    = battle.start()
    response = _decode_obs(raw)
 
    if response is None:
        return None, StartData(error="BattleStart returned empty response", player0=0, player1=1)
 
    error = response.get("error")
    if error:
        return None, StartData(error=error, player0=0, player1=1)
 
    start_data = StartData(
        error   = None,
        player0 = response.get("player0", 0),
        player1 = response.get("player1", 1),
    )
 
    # The initial observation is nested under "observation" or is the top-level dict
    obs_dict = response.get("observation", response)
 
    # Carry search_begin_input forward if the engine embeds it here
    sbi = response.get("search_begin_input")
    if sbi is not None and isinstance(obs_dict, dict):
        obs_dict["search_begin_input"] = sbi
 
    # Stash the Battle instance so battle_select / battle_finish can reach it
    _set_active_battle(battle, obs_dict)
 
    return obs_dict, start_data
 