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
 