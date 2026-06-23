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

# Load all card data from the API's helper function
all_card = all_card_data()
# Create a lookup table (dictionary) to quickly access card data by its cardId
card_table = {c.cardId:c for c in all_card}
card_count = max(all_card, key=lambda c: c.cardId).cardId + 1 # Max Card ID + 1

attack_count = max(all_attack(), key=lambda a: a.attackId).attackId + 1 # Max Attack ID + 1

num_words_encoder = 24
encoder_size = 22000 # Encoder input size exceeding the vocabulary size

decoder_main_feature = 8 # Feature count of SelectContext.Main
decoder_attack_offset = 14 # First index of Attack feature
decoder_card_offset = decoder_attack_offset + attack_count # First index of Card Feature
decoder_size = decoder_card_offset + (1 + decoder_main_feature + SelectContext.RECOVER_SPECIAL_CONDITION) * card_count # Decoder input vocabulary size

SEARCH_COUNT = 10
