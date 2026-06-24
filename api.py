"""
api.py — Python enums, dataclasses, and high-level API functions for the cabt engine.

Mirrors the full API reference at https://matsuoinstitute.github.io/cabt/api.html

The C++ core is accessed through sim.py (ctypes bindings to libcg.so / cg.dll).
This module builds typed Python objects on top of those raw JSON strings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AreaType(IntEnum):
    DECK          = 0
    HAND          = 1
    DISCARD       = 2
    ACTIVE        = 3
    BENCH         = 4
    PRIZE         = 5
    STADIUM       = 6
    ENERGY        = 7
    TOOL          = 8
    PRE_EVOLUTION = 9
    PLAYER        = 10
    LOOKING       = 11


class EnergyType(IntEnum):
    COLORLESS   = 0
    GRASS       = 1
    FIRE        = 2
    WATER       = 3
    LIGHTNING   = 4
    PSYCHIC     = 5
    FIGHTING    = 6
    DARKNESS    = 7
    METAL       = 8
    DRAGON      = 9
    RAINBOW     = 10
    TEAM_ROCKET = 11


class CardType(IntEnum):
    POKEMON        = 0
    ITEM           = 1
    TOOL           = 2
    SUPPORTER      = 3
    STADIUM        = 4
    BASIC_ENERGY   = 5
    SPECIAL_ENERGY = 6


class SpecialConditionType(IntEnum):
    POISON   = 0
    BURN     = 1
    SLEEP    = 2
    PARALYZE = 3
    CONFUSE  = 4


class SelectType(IntEnum):
    MAIN                = 0
    CARD                = 1
    ATTACHED_CARD       = 2
    CARD_OR_ATTACHED_CARD = 3
    ENERGY              = 4
    SKILL               = 5
    ATTACK              = 6
    EVOLVE              = 7
    COUNT               = 8
    YES_NO              = 9
    SPECIAL_CONDITION   = 10


class SelectContext(IntEnum):
    MAIN                        = 0
    SETUP_ACTIVE_POKEMON        = 1
    SETUP_BENCH_POKEMON         = 2
    SWITCH                      = 3
    TO_ACTIVE                   = 4
    TO_BENCH                    = 5
    TO_FIELD                    = 6
    TO_HAND                     = 7
    DISCARD                     = 8
    TO_DECK                     = 9
    TO_DECK_BOTTOM              = 10
    TO_PRIZE                    = 11
    NOT_MOVE                    = 12
    DAMAGE_COUNTER              = 13
    DAMAGE_COUNTER_ANY          = 14
    DAMAGE                      = 15
    REMOVE_DAMAGE_COUNTER       = 16
    HEAL                        = 17
    EVOLVES_FROM                = 18
    EVOLVES_TO                  = 19
    DEVOLVE                     = 20
    ATTACH_FROM                 = 21
    ATTACH_TO                   = 22
    DETACH_FROM                 = 23
    LOOK                        = 24
    EFFECT_TARGET               = 25
    DISCARD_ENERGY_CARD         = 26
    DISCARD_TOOL_CARD           = 27
    SWITCH_ENERGY_CARD          = 28
    DISCARD_CARD_OR_ATTACHED_CARD = 29
    DISCARD_ENERGY              = 30
    TO_HAND_ENERGY              = 31
    TO_DECK_ENERGY              = 32
    SWITCH_ENERGY               = 33
    SKILL_ORDER                 = 34
    ATTACK                      = 35
    DISABLE_ATTACK              = 36
    EVOLVE                      = 37
    DRAW_COUNT                  = 38
    DAMAGE_COUNTER_COUNT        = 39
    REMOVE_DAMAGE_COUNTER_COUNT = 40
    IS_FIRST                    = 41
    MULLIGAN                    = 42
    ACTIVATE                    = 43
    FIRST_EFFECT                = 44
    MORE_DEVOLVE                = 45
    COIN_HEAD                   = 46
    AFFECT_SPECIAL_CONDITION    = 47
    RECOVER_SPECIAL_CONDITION   = 48


class OptionType(IntEnum):
    NUMBER           = 0
    YES              = 1
    NO               = 2
    CARD             = 3
    TOOL_CARD        = 4
    ENERGY_CARD      = 5
    ENERGY           = 6
    PLAY             = 7
    ATTACH           = 8
    EVOLVE           = 9
    ABILITY          = 10
    DISCARD          = 11
    RETREAT          = 12
    ATTACK           = 13
    END              = 14
    SKILL            = 15
    SPECIAL_CONDITION = 16


class LogType(IntEnum):
    SHUFFLE          = 0
    HAS_BASIC_POKEMON = 1
    TURN_START       = 2
    TURN_END         = 3
    DRAW             = 4
    DRAW_REVERSE     = 5
    MOVE_CARD        = 6
    MOVE_CARD_REVERSE = 7
    SWITCH           = 8
    CHANGE           = 9
    PLAY             = 10
    ATTACH           = 11
    EVOLVE           = 12
    DEVOLVE          = 13
    MOVE_ATTACHED    = 14
    ATTACK           = 15
    HP_CHANGE        = 16
    POISONED         = 17
    BURNED           = 18
    ASLEEP           = 19
    PARALYZED        = 20
    CONFUSED         = 21
    COIN             = 22
    RESULT           = 23


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Card:
    """A card reference on the board (not its full metadata)."""
    id:          int            # Card-type ID (matches all_card_data keys)
    playerIndex: int            # Which player owns this card (0 or 1)
    serial:      int            # Unique instance ID within the battle

    @classmethod
    def from_dict(cls, d: dict) -> "Card":
        return cls(
            id          = d["id"],
            playerIndex = d["playerIndex"],
            serial      = d["serial"],
        )


@dataclass
class Pokemon:
    """A Pokémon on the board (active or bench slot)."""
    id:             int                    # Card-type ID
    serial:         int                    # Unique instance ID
    hp:             int                    # Current HP
    maxHp:          int                    # Max HP
    appearThisTurn: bool                   # Placed or evolved this turn (can't retreat/attack)
    energies:       list[EnergyType]       # Energy types currently attached (for cost checks)
    energyCards:    list[Card]             # The actual energy card objects attached
    tools:          list[Card]             # Tool cards attached
    preEvolution:   Optional["Pokemon"]    # The Pokémon this evolved from (if any)

    @classmethod
    def from_dict(cls, d: dict) -> "Pokemon":
        if d is None:
            return None
        pre = d.get("preEvolution")
        return cls(
            id             = d["id"],
            serial         = d["serial"],
            hp             = d["hp"],
            maxHp          = d["maxHp"],
            appearThisTurn = d.get("appearThisTurn", False),
            energies       = [EnergyType(e) for e in d.get("energies", [])],
            energyCards    = [Card.from_dict(c) for c in d.get("energyCards", [])],
            tools          = [Card.from_dict(c) for c in d.get("tools", [])],
            preEvolution   = Pokemon.from_dict(pre) if pre else None,
        )


@dataclass
class PlayerState:
    """Everything visible about one player's board side."""
    # Active slot: list of 0 or 1 entries; element may be None if face-down
    active:    list[Optional[Pokemon]]
    # Bench: up to benchMax Pokémon
    bench:     list[Pokemon]
    benchMax:  int
    # Hand: visible only for the player whose perspective we have
    hand:      list[Card]
    handCount: int            # Always available (even for opponent)
    deckCount: int
    # Prize cards: None entries are face-down (not yet revealed)
    prize:     list[Optional[Card]]
    # Discard pile (fully visible to both players)
    discard:   list[Card]
    # Special conditions
    poisoned:  bool
    burned:    bool
    asleep:    bool
    paralyzed: bool
    confused:  bool

    @classmethod
    def from_dict(cls, d: dict) -> "PlayerState":
        active_raw = d.get("active", [])
        # Each slot is either a Pokemon dict or None
        active = [
            Pokemon.from_dict(p) if p is not None else None
            for p in active_raw
        ]
        bench = [Pokemon.from_dict(p) for p in d.get("bench", []) if p is not None]
        prize = [
            Card.from_dict(p) if p is not None else None
            for p in d.get("prize", [])
        ]
        return cls(
            active    = active,
            bench     = bench,
            benchMax  = d.get("benchMax", 5),
            hand      = [Card.from_dict(c) for c in d.get("hand", [])],
            handCount = d.get("handCount", 0),
            deckCount = d.get("deckCount", 0),
            prize     = prize,
            discard   = [Card.from_dict(c) for c in d.get("discard", [])],
            poisoned  = d.get("poisoned",  False),
            burned    = d.get("burned",    False),
            asleep    = d.get("asleep",    False),
            paralyzed = d.get("paralyzed", False),
            confused  = d.get("confused",  False),
        )


@dataclass
class State:
    """Complete board state at a given game moment."""
    players:         list[PlayerState]    # Index 0 and 1
    yourIndex:       int                  # Which player this observation is addressed to
    firstPlayer:     int                  # Who went first (0 or 1)
    turn:            int                  # Current turn number (1-indexed)
    turnActionCount: int                  # Number of actions taken this turn
    supporterPlayed: bool
    stadiumPlayed:   bool
    energyAttached:  bool
    retreated:       bool
    stadium:         Optional[Card]       # Currently active stadium card
    looking:         Any                  # Cards currently being "looked at" (search effects)
    result:          Optional[int]        # None = ongoing; 0/1 = that player won; -1 = draw

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        if d is None:
            return None
        stadium_raw = d.get("stadium")
        return cls(
            players         = [PlayerState.from_dict(p) for p in d["players"]],
            yourIndex       = d.get("yourIndex", 0),
            firstPlayer     = d.get("firstPlayer", 0),
            turn            = d.get("turn", 0),
            turnActionCount = d.get("turnActionCount", 0),
            supporterPlayed = d.get("supporterPlayed", False),
            stadiumPlayed   = d.get("stadiumPlayed",   False),
            energyAttached  = d.get("energyAttached",  False),
            retreated       = d.get("retreated",       False),
            stadium         = Card.from_dict(stadium_raw) if stadium_raw else None,
            looking         = d.get("looking"),
            result          = d.get("result"),
        )


@dataclass
class Option:
    """One selectable action in the current game state."""
    type:                 OptionType
    # Fields are populated depending on type; unused fields are None
    index:                Optional[int]                  = None  # Generic positional index
    serial:               Optional[int]                  = None  # Instance serial of the card
    cardId:               Optional[int]                  = None  # Card-type ID
    playerIndex:          Optional[int]                  = None  # Owning player
    area:                 Optional[AreaType]              = None  # Where the card is
    inPlayArea:           Optional[AreaType]              = None  # In-play zone
    inPlayIndex:          Optional[int]                  = None  # Slot within zone
    attackId:             Optional[int]                  = None  # Attack identifier
    energyIndex:          Optional[int]                  = None  # Energy slot index
    toolIndex:            Optional[int]                  = None  # Tool slot index
    count:                Optional[int]                  = None  # Count for COUNT selects
    number:               Optional[int]                  = None  # Numeric value
    specialConditionType: Optional[SpecialConditionType] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Option":
        def _opt_enum(cls_, key):
            v = d.get(key)
            return cls_(v) if v is not None else None

        return cls(
            type                 = OptionType(d["type"]),
            index                = d.get("index"),
            serial               = d.get("serial"),
            cardId               = d.get("cardId"),
            playerIndex          = d.get("playerIndex"),
            area                 = _opt_enum(AreaType,              "area"),
            inPlayArea           = _opt_enum(AreaType,              "inPlayArea"),
            inPlayIndex          = d.get("inPlayIndex"),
            attackId             = d.get("attackId"),
            energyIndex          = d.get("energyIndex"),
            toolIndex            = d.get("toolIndex"),
            count                = d.get("count"),
            number               = d.get("number"),
            specialConditionType = _opt_enum(SpecialConditionType,  "specialConditionType"),
        )


@dataclass
class SelectData:
    """Describes the current selection the active player must make."""
    type:                SelectType
    context:             SelectContext
    option:              list[Option]
    minCount:            int
    maxCount:            int
    contextCard:         Optional[Card]  = None  # Card that triggered this select
    deck:                Optional[list[Card]] = None  # Visible deck for look effects
    effect:              Optional[str]   = None  # Text description of the effect
    remainDamageCounter: Optional[int]  = None
    remainEnergyCost:    Optional[list[EnergyType]] = None

    @classmethod
    def from_dict(cls, d: dict) -> "SelectData":
        if d is None:
            return None
        ctx_card_raw = d.get("contextCard")
        deck_raw     = d.get("deck")
        remain_cost  = d.get("remainEnergyCost")
        return cls(
            type                = SelectType(d["type"]),
            context             = SelectContext(d["context"]),
            option              = [Option.from_dict(o) for o in d.get("option", [])],
            minCount            = d.get("minCount", 1),
            maxCount            = d.get("maxCount", 1),
            contextCard         = Card.from_dict(ctx_card_raw) if ctx_card_raw else None,
            deck                = [Card.from_dict(c) for c in deck_raw] if deck_raw else None,
            effect              = d.get("effect"),
            remainDamageCounter = d.get("remainDamageCounter"),
            remainEnergyCost    = [EnergyType(e) for e in remain_cost] if remain_cost else None,
        )


@dataclass
class Log:
    """A single game event log entry."""
    type: LogType
    # Additional payload varies by log type; stored as raw dict for flexibility
    data: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Log":
        log_type = LogType(d["type"])
        return cls(type=log_type, data={k: v for k, v in d.items() if k != "type"})


@dataclass
class Observation:
    """
    Top-level object returned by the engine after every selection.

    This is what the cabt engine calls obs_dict at the Python level.
    """
    logs:               list[Log]
    current:            Optional[State]       # None during initial deck-pick phase
    select:             Optional[SelectData]  # None when no selection is needed
    search_begin_input: Optional[Any]         # Payload for search_begin() calls

    @classmethod
    def from_dict(cls, d: dict) -> "Observation":
        current_raw = d.get("current")
        select_raw  = d.get("select")
        return cls(
            logs               = [Log.from_dict(lg) for lg in d.get("logs", [])],
            current            = State.from_dict(current_raw) if current_raw else None,
            select             = SelectData.from_dict(select_raw) if select_raw else None,
            search_begin_input = d.get("search_begin_input"),
        )


@dataclass
class SearchState:
    """Returned by search_begin(); used for lookahead without committing actions."""
    searchId:    int
    observation: Observation

    @classmethod
    def from_dict(cls, d: dict) -> "SearchState":
        return cls(
            searchId    = d["searchId"],
            observation = Observation.from_dict(d["observation"]),
        )


@dataclass
class ApiResult:
    """Generic wrapper around engine responses that can carry an error."""
    state: Optional[Observation]
    error: Optional[str]

    @classmethod
    def from_dict(cls, d: dict) -> "ApiResult":
        state_raw = d.get("state")
        return cls(
            state = Observation.from_dict(state_raw) if state_raw else None,
            error = d.get("error"),
        )


# ---------------------------------------------------------------------------
# Card metadata dataclasses (returned by all_card_data / all_attack)
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    """An ability / Poké-Power / Poké-Body on a card."""
    name: str
    text: str

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        return cls(name=d["name"], text=d.get("text", ""))


@dataclass
class Attack:
    """An attack belonging to a Pokémon card."""
    attackId: int
    name:     str
    text:     str
    damage:   int                    # Base damage (0 if variable)
    energies: list[EnergyType]       # Required energy cost

    @classmethod
    def from_dict(cls, d: dict) -> "Attack":
        return cls(
            attackId = d["attackId"],
            name     = d["name"],
            text     = d.get("text", ""),
            damage   = d.get("damage", 0),
            energies = [EnergyType(e) for e in d.get("energies", [])],
        )


@dataclass
class CardData:
    """Full metadata for one card type (returned by all_card_data)."""
    cardId:      int
    name:        str
    cardType:    CardType
    hp:          int                     # 0 for non-Pokémon
    energyType:  Optional[EnergyType]   # Type of the Pokémon / energy card
    evolvesFrom: Optional[str]          # Name of the pre-evolution, if any
    retreatCost: int                     # Number of energy to retreat
    weakness:    Optional[EnergyType]
    resistance:  Optional[EnergyType]
    attacks:     list[Attack]
    skills:      list[Skill]
    # Stage / variant flags
    basic:   bool = False
    stage1:  bool = False
    stage2:  bool = False
    ex:      bool = False
    megaEx:  bool = False
    tera:    bool = False
    aceSpec: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "CardData":
        def _opt_energy(key):
            v = d.get(key)
            return EnergyType(v) if v is not None else None

        return cls(
            cardId      = d["cardId"],
            name        = d["name"],
            cardType    = CardType(d["cardType"]),
            hp          = d.get("hp", 0),
            energyType  = _opt_energy("energyType"),
            evolvesFrom = d.get("evolvesFrom"),
            retreatCost = d.get("retreatCost", 0),
            weakness    = _opt_energy("weakness"),
            resistance  = _opt_energy("resistance"),
            attacks     = [Attack.from_dict(a) for a in d.get("attacks", [])],
            skills      = [Skill.from_dict(s) for s in d.get("skills", [])],
            basic       = d.get("basic",   False),
            stage1      = d.get("stage1",  False),
            stage2      = d.get("stage2",  False),
            ex          = d.get("ex",      False),
            megaEx      = d.get("megaEx",  False),
            tera        = d.get("tera",    False),
            aceSpec     = d.get("aceSpec", False),
        )


# ---------------------------------------------------------------------------
# Module-level API functions
# ---------------------------------------------------------------------------

# Cached after first call (card data never changes during a run)
_card_cache:   Optional[dict[int, CardData]] = None
_attack_cache: Optional[dict[int, Attack]]   = None


def all_card_data() -> dict[int, CardData]:
    """
    Return metadata for every card in the competition card pool.

    Returns a dict keyed by cardId.  Cached after the first call.
    """
    global _card_cache
    if _card_cache is not None:
        return _card_cache

    import sim  #
    raw_json: str = sim.AllCard()
    data: list[dict] = json.loads(raw_json)
    _card_cache = {d["cardId"]: CardData.from_dict(d) for d in data}
    return _card_cache


def all_attack() -> dict[int, Attack]:
    """
    Return metadata for every attack in the card pool.

    Returns a dict keyed by attackId.  Cached after the first call.
    """
    global _attack_cache
    if _attack_cache is not None:
        return _attack_cache

    import sim
    raw_json: str = sim.AllAttack()
    data: list[dict] = json.loads(raw_json)
    _attack_cache = {d["attackId"]: Attack.from_dict(d) for d in data}
    return _attack_cache


def to_observation_class(obs_dict: dict) -> Observation:
    """
    Convert a raw obs_dict (as returned by game.battle_start / battle_select)
    into typed Python dataclasses.

    Use this when you want attribute access instead of dict indexing:

        obs = to_observation_class(raw)
        hp  = obs.current.players[0].active[0].hp
    """
    return Observation.from_dict(obs_dict)


def search_begin(obs_dict: dict) -> SearchState:
    """
    Initialise a lookahead search from the current observation.

    Does NOT commit any actions.  Use search_step() to explore branches and
    search_end() / search_release() to clean up when done.

    Returns a SearchState containing the search ID and a copy of the current
    observation annotated for search use.
    """
    import sim
    input_data = obs_dict.get("search_begin_input", obs_dict)
    raw_json   = sim.SearchBegin(json.dumps(input_data).encode())
    result     = json.loads(raw_json)
    return SearchState.from_dict(result)


def search_step(search_id: int, select: list[int]) -> SearchState:
    """
    Advance a lookahead search by one step with the given option indices.

    Parameters
    ----------
    search_id:
        The ID returned by search_begin().
    select:
        List of option indices (same format as game.battle_select).

    Returns the updated SearchState after applying the selection.
    """
    import sim
    payload  = json.dumps({"searchId": search_id, "select": select}).encode()
    raw_json = sim.SearchStep(payload)
    result   = json.loads(raw_json)
    return SearchState.from_dict(result)


def search_end() -> None:
    """End the current search session and release all resources."""
    import sim
    sim.SearchEnd()


def search_release(search_id: int) -> None:
    """Explicitly destroy a specific search branch by ID."""
    import sim
    payload = json.dumps({"searchId": search_id}).encode()
    sim.SearchRelease(payload)