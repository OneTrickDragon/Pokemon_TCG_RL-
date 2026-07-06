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
    vec[i] = min(current.get("turn", 0), MAX_TURNS) / MAX_TURNS;  i += 1
    vec[i] = float(player_index);                            i += 1
    vec[i] = float(current.get("firstPlayer", 0));           i += 1
    vec[i] = float(current.get("supporterPlayed", False));   i += 1
    vec[i] = float(current.get("stadiumPlayed",   False));   i += 1
    vec[i] = float(current.get("energyAttached",  False));   i += 1
    vec[i] = float(current.get("retreated",       False));   i += 1
 
    # Prize counts
    me_prizes  = min(len(me.get("prize", [])), MAX_PRIZE)
    opp_prizes = min(len(opp.get("prize", [])), MAX_PRIZE)
    vec[i] = me_prizes  / MAX_PRIZE;  i += 1
    vec[i] = opp_prizes / MAX_PRIZE;  i += 1
 
    # Deck + hand counts
    vec[i] = me.get("deckCount",  0) / 60.0;  i += 1
    vec[i] = opp.get("deckCount", 0) / 60.0;  i += 1
    vec[i] = min(me.get("handCount",  0), MAX_HAND) / MAX_HAND;  i += 1
    vec[i] = min(opp.get("handCount", 0), MAX_HAND) / MAX_HAND;  i += 1
 
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
    """Number of prize cards remaining, whether face-down or revealed."""
    return len(player_state.get("prize", []))
 
 
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


def _result_from_obs(obs_dict: dict | None) -> int | None:
    current = (obs_dict or {}).get("current")
    if current is None:
        return None
    return current.get("result")


def _option_count(obs_dict: dict | None) -> int:
    select = (obs_dict or {}).get("select") or {}
    return len(select.get("option", []))


def _max_select_count(obs_dict: dict | None) -> int:
    select = (obs_dict or {}).get("select") or {}
    return int(select.get("maxCount", 0) or 0)


def _build_select_list(obs_dict: dict, action: int, rng: random.Random | None = None) -> list[int]:
    """
    Convert one discrete action into the list format expected by cabt.

    The first selected index comes from the policy. Any additional required
    choices are filled uniformly from remaining legal options.
    """
    rng = rng or random
    n_options = _option_count(obs_dict)
    max_count = _max_select_count(obs_dict)
    if n_options == 0 or max_count <= 0:
        return []

    first = int(action) % n_options
    selected = [first]
    if max_count == 1:
        return selected

    remaining = [i for i in range(n_options) if i != first]
    rng.shuffle(remaining)
    selected.extend(remaining[: max_count - 1])
    return selected[:max_count]


def random_agent(obs_dict: dict) -> list[int]:
    """Uniform random baseline that always returns a legal selection shape."""
    n_options = _option_count(obs_dict)
    if n_options == 0:
        return []
    return _build_select_list(obs_dict, random.randrange(n_options))


def greedy_agent(obs_dict: dict) -> list[int]:
    """
    Simple rule-based opponent.

    Priority order: ATTACK > EVOLVE > PLAY > ATTACH > ABILITY > RETREAT > END.
    """
    select = obs_dict.get("select") or {}
    options = select.get("option", [])
    if not options:
        return []

    priority = {
        OptionType.ATTACK: 0,
        OptionType.EVOLVE: 1,
        OptionType.PLAY: 2,
        OptionType.ATTACH: 3,
        OptionType.ABILITY: 4,
        OptionType.RETREAT: 5,
        OptionType.END: 6,
    }

    best_index = 0
    best_score = 99
    for i, option in enumerate(options):
        option_type = option.get("type")
        try:
            option_type = OptionType(option_type)
        except (TypeError, ValueError):
            pass
        score = priority.get(option_type, 50)
        if score < best_score:
            best_index = i
            best_score = score
    return _build_select_list(obs_dict, best_index)


class PTCGEnv(gym.Env):
    """
    Single-agent Gym wrapper around one cabt battle.

    The learning agent controls `player_index`; the other player is advanced
    internally with `opponent_agent`.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        deck: list[int],
        opponent_deck: list[int] | None = None,
        player_index: int = 0,
        opponent_agent=random_agent,
        max_turns: int = MAX_TURNS,
    ) -> None:
        if not HAS_GYM:
            raise ImportError("PTCGEnv requires gymnasium or gym to be installed.")
        if player_index not in (0, 1):
            raise ValueError("player_index must be 0 or 1")
        if len(deck) != 60:
            raise ValueError(f"deck must have 60 cards, got {len(deck)}")
        if opponent_deck is not None and len(opponent_deck) != 60:
            raise ValueError(f"opponent_deck must have 60 cards, got {len(opponent_deck)}")

        self.deck = list(deck)
        self.opponent_deck = list(opponent_deck or deck)
        self.player_index = player_index
        self.opponent_agent = opponent_agent
        self.max_turns = max_turns

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(MAX_OPTIONS)

        self._obs_dict: dict | None = None
        self._done = False
        self._last_reward_obs: dict | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self._obs_dict is not None:
            battle_finish()

        deck0 = self.deck if self.player_index == 0 else self.opponent_deck
        deck1 = self.opponent_deck if self.player_index == 0 else self.deck
        obs_dict, start_data = battle_start(deck0, deck1)
        if start_data.error:
            raise RuntimeError(f"battle_start failed: {start_data.error}")

        self._obs_dict = obs_dict
        self._done = False
        self._process_until_our_turn()
        self._last_reward_obs = self._obs_dict
        return self._encoded_obs(), self._info()

    def step(self, action: int):
        if self._obs_dict is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        prev_obs = self._last_reward_obs
        select_list = _build_select_list(self._obs_dict, int(action), self.np_random)
        self._obs_dict = battle_select(select_list)
        self._process_until_our_turn()

        result = _result_from_obs(self._obs_dict)
        terminated = result is not None
        truncated = self._is_truncated()
        self._done = terminated or truncated

        reward = compute_reward(prev_obs, self._obs_dict, self.player_index, terminated, result)
        self._last_reward_obs = self._obs_dict

        if self._done:
            battle_finish()

        return self._encoded_obs(), reward, terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(MAX_OPTIONS, dtype=bool)
        n_options = min(_option_count(self._obs_dict), MAX_OPTIONS)
        mask[:n_options] = True
        return mask

    def render(self):
        from game import visualize_data

        if self._obs_dict is None:
            return ""
        return visualize_data()

    def close(self) -> None:
        if self._obs_dict is not None and not self._done:
            battle_finish()
        self._obs_dict = None
        self._done = True

    def _process_until_our_turn(self) -> None:
        while self._obs_dict is not None:
            result = _result_from_obs(self._obs_dict)
            if result is not None or self._is_truncated():
                return

            select = self._obs_dict.get("select")
            if select is None:
                self._obs_dict = battle_select([])
                continue

            current = self._obs_dict.get("current")
            if current is None or current.get("yourIndex") == self.player_index:
                return

            opponent_select = self.opponent_agent(self._obs_dict)
            self._obs_dict = battle_select(opponent_select)

    def _encoded_obs(self) -> np.ndarray:
        return encode_observation(self._obs_dict or {}, self.player_index)

    def _is_truncated(self) -> bool:
        current = (self._obs_dict or {}).get("current")
        if current is None:
            return False
        return current.get("turn", 0) >= self.max_turns and current.get("result") is None

    def _info(self) -> dict[str, Any]:
        return {
            "obs_dict": self._obs_dict,
            "action_mask": self.action_masks(),
            "player_index": self.player_index,
            "result": _result_from_obs(self._obs_dict),
        }


CabtEnv = PTCGEnv
