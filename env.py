from __future__ import annotations
 
import random
from typing import Any
 
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        HAS_GYM = True
    except ImportError:
        HAS_GYM = False
        # Provide stub so type hints don't break at import time
        class _Stub:
            Env = object
        gym = _Stub()
        spaces = None


from game import battle_start, battle_select, battle_finish  # noqa: E402
from api import (  # noqa: E402
    OptionType,
    SelectType,
    SelectContext,
    CardType,
    EnergyType,
)
 
MAX_BENCH        = 5
MAX_HAND         = 10   # practical cap for feature vector; hands rarely exceed this
MAX_OPTIONS      = 64   # practical cap on the legal-move list
MAX_PRIZE        = 6
MAX_HP           = 400  # highest HP on any current card
MAX_ENERGY_TYPES = 12   # number of EnergyType enum values
MAX_TURNS        = 200 

 
OBS_DIM = 61

def _encode_pokemon(poke) -> list[float]:
    """Return 4 floats for a Pokémon slot (or zeros if empty)."""
    if poke is None:
        return [0.0, 0.0, 0.0, 0.0]
    hp_ratio   = poke["hp"] / max(poke["maxHp"], 1)
    energies   = min(len(poke.get("energies", [])), 10) / 10.0
    tools      = min(len(poke.get("tools", [])), 4) / 4.0
    appeared   = float(poke.get("appearThisTurn", False))
    return [hp_ratio, energies, tools, appeared]


def encode_observation(obs_dict: dict, player_index: int) -> np.ndarray:
    """
    Convert a raw cabt obs_dict into a fixed-size float32 numpy vector.
 
    player_index: which player we are (0 or 1) — used to orient self/opp.
    """
    vec = np.zeros(OBS_DIM, dtype=np.float32)
 
    current = obs_dict.get("current")
    if current is None:
        # Setup / mulligan phase — no board yet
        return vec
 
    opp_index = 1 - player_index
    me  = current["players"][player_index]
    opp = current["players"][opp_index]
 
    i = 0
    vec[i] = current.get("turn", 0) / MAX_TURNS;            i += 1
    vec[i] = float(player_index);                            i += 1
    vec[i] = float(current.get("firstPlayer", 0));           i += 1
    vec[i] = float(current.get("supporterPlayed", False));   i += 1
    vec[i] = float(current.get("stadiumPlayed",   False));   i += 1
    vec[i] = float(current.get("energyAttached",  False));   i += 1
    vec[i] = float(current.get("retreated",       False));   i += 1
 
    # Prize counts
    me_prizes  = sum(1 for p in me.get("prize",  []) if p is not None)
    opp_prizes = sum(1 for p in opp.get("prize", []) if p is not None)
    vec[i] = me_prizes  / MAX_PRIZE;  i += 1
    vec[i] = opp_prizes / MAX_PRIZE;  i += 1
 
    # Deck + hand counts
    vec[i] = me.get("deckCount",  0) / 60.0;  i += 1
    vec[i] = opp.get("deckCount", 0) / 60.0;  i += 1
    vec[i] = me.get("handCount",  0) / MAX_HAND;  i += 1
    vec[i] = opp.get("handCount", 0) / MAX_HAND;  i += 1
 
    # Active Pokémon
    me_active  = (me.get("active")  or [None])[0]
    opp_active = (opp.get("active") or [None])[0]
    for val in _encode_pokemon(me_active):
        vec[i] = val;  i += 1
    for val in _encode_pokemon(opp_active):
        vec[i] = val;  i += 1
 
    # Bench (up to MAX_BENCH slots each)
    for bench_list in [me.get("bench", []), opp.get("bench", [])]:
        for slot in range(MAX_BENCH):
            poke = bench_list[slot] if slot < len(bench_list) else None
            for val in _encode_pokemon(poke):
                vec[i] = val;  i += 1
 
    assert i == OBS_DIM, f"encode_observation produced {i} values, expected {OBS_DIM}"
    return vec

 
def _prize_count(player_state) -> int:
    """Number of face-up prize cards remaining (i.e. not yet taken)."""
    return sum(1 for p in player_state.get("prize", []) if p is not None)
 
 
def compute_reward(
    prev_obs: dict | None,
    new_obs:  dict | None,
    player_index: int,
    done: bool,
    result: int | None,
) -> float:
    """
    Shaped reward from the perspective of `player_index`.
 
    Terminal:
      +1.0  win,  -1.0  loss,  0.0  draw / unknown
 
    Per-step shaping (small signals to reduce sparsity):
      +0.1   for each prize card WE took this step
      -0.1   for each prize card OPP took this step
      +0.05  for KO'ing opponent's active (active disappeared and opp prizes went down)
      -0.05  for our active being KO'd
    """
    reward = 0.0
 
    if done and result is not None:
        if result == player_index:
            reward += 1.0
        elif result == (1 - player_index):
            reward -= 1.0
        # result == -1  → draw
        return reward
 
    if prev_obs is None or new_obs is None:
        return reward
 
    prev_curr = prev_obs.get("current")
    new_curr  = new_obs.get("current")
    if prev_curr is None or new_curr is None:
        return reward
 
    opp_index = 1 - player_index
    prev_me   = prev_curr["players"][player_index]
    prev_opp  = prev_curr["players"][opp_index]
    new_me    = new_curr["players"][player_index]
    new_opp   = new_curr["players"][opp_index]
 
    # Prizes taken = previous prize count - current prize count
    # (lower remaining prizes = more taken by the other side)
    my_prizes_taken  = _prize_count(prev_opp) - _prize_count(new_opp)
    opp_prizes_taken = _prize_count(prev_me)  - _prize_count(new_me)
 
    reward += 0.1 * my_prizes_taken
    reward -= 0.1 * opp_prizes_taken
 
    return reward