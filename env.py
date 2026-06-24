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