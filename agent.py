import glob
import math
import os
import random
import sys

import torch
import torch.nn
import torch.nn.functional
import torch.optim

sys.path.append(glob.glob('/kaggle/input/**/cg-lib', recursive=True)[0])

from cg.api import (
    AreaType,
    Card,
    Observation,
    OptionType,
    PlayerState,
    Pokemon,
    SearchState,
    SelectContext,
    all_attack,
    all_card_data,
    search_begin,
    search_end,
    search_step,
    to_observation_class,
)
from cg.game import battle_start, battle_finish, battle_select