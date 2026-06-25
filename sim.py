"""
sim.py — Low-level interface to the cabt C++ shared library (libcg.so / cg.dll).

Mirrors the sim module described at https://matsuoinstitute.github.io/cabt/sim.html

Module-level globals (as specified in the docs):
  lib            — the loaded ctypes.CDLL instance
  GameInitialize — bound C function
  BattleStart    — bound C function
  AgentStart     — bound C function
  BattleFinish   — bound C function
  GetBattleData  — bound C function
  Select         — bound C function
  VisualizeData  — bound C function
  SearchBegin    — bound C function
  SearchStep     — bound C function
  SearchEnd      — bound C function
  SearchRelease  — bound C function
  AllCard        — bound C function
  AllAttack      — bound C function

Classes:
  StartData   — result of BattleStart; carries player assignments and error info
  SerialData  — maps card serial numbers to card-type IDs
  Battle      — high-level wrapper around one battle session (owns the C state)

All C functions communicate through null-terminated UTF-8 JSON strings so no
additional serialisation layer is needed beyond json.loads / json.dumps.

Calling convention assumed (cabt SDK v0.1.0):
  void  GameInitialize()
  char* BattleStart(const char* deck0_json, const char* deck1_json)
  char* AgentStart(int player_index)
  void  BattleFinish()
  char* GetBattleData(int player_index)
  char* Select(const char* select_json)
  char* VisualizeData()
  char* SearchBegin(const char* input_json)
  char* SearchStep(const char* step_json)
  void  SearchEnd()
  void  SearchRelease(const char* release_json)
  char* AllCard()
  char* AllAttack()
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
from typing import Optional


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _find_library() -> str:
    """
    Return the path to the cabt shared library.

    Search order:
      1. CABT_LIB env var — full path to the .so / .dll
      2. Directory containing this file
      3. Current working directory
      4. Bare name (OS loader searches LD_LIBRARY_PATH / PATH / etc.)
    """
    lib_name = "cg.dll" if platform.system() == "Windows" else "libcg.so"

    env = os.environ.get("CABT_LIB")
    if env:
        return env

    here = os.path.dirname(os.path.abspath(__file__))
    for directory in (here, os.getcwd()):
        candidate = os.path.join(directory, lib_name)
        if os.path.isfile(candidate):
            return candidate

    return lib_name  # last resort: let the OS loader find it


def _load_lib() -> ctypes.CDLL:
    path = _find_library()
    try:
        return ctypes.CDLL(path)
    except OSError as exc:
        raise FileNotFoundError(
            f"Could not load the cabt shared library at '{path}'.\n"
            "Place libcg.so (or cg.dll) next to sim.py, or set the "
            "CABT_LIB environment variable to its full path."
        ) from exc


# ---------------------------------------------------------------------------
# ctypes helpers
# ---------------------------------------------------------------------------

_c_str  = ctypes.c_char_p
_c_int  = ctypes.c_int
_c_void = None   # restype sentinel for void


def _bind(lib: ctypes.CDLL, name: str, argtypes: list, restype) -> ctypes.CFUNCTYPE:
    """
    Configure and return a single bound C function.

    Setting argtypes / restype explicitly prevents silent integer truncation
    on 64-bit platforms and enables automatic None-for-NULL handling on
    char* returns.
    """
    fn          = getattr(lib, name)
    fn.argtypes = argtypes or None
    fn.restype  = restype
    return fn


# ---------------------------------------------------------------------------
# Module-level globals  (as listed in the official docs)
# ---------------------------------------------------------------------------

#: The loaded C++ shared library.  Available for introspection / direct calls.
lib: ctypes.CDLL = _load_lib()

# Bind every exported symbol at import time so that attribute access is fast
# and errors surface immediately (not on first call deep inside a training run).

#: void GameInitialize()
GameInitialize: ctypes.CFUNCTYPE = _bind(lib, "GameInitialize", [], _c_void)

#: char* BattleStart(const char* deck0_json, const char* deck1_json)
BattleStart:    ctypes.CFUNCTYPE = _bind(lib, "BattleStart",   [_c_str, _c_str], _c_str)

#: char* AgentStart(int player_index)
AgentStart:     ctypes.CFUNCTYPE = _bind(lib, "AgentStart",    [_c_int],          _c_str)

#: void BattleFinish()
BattleFinish:   ctypes.CFUNCTYPE = _bind(lib, "BattleFinish",  [],                _c_void)

#: char* GetBattleData(int player_index)
GetBattleData:  ctypes.CFUNCTYPE = _bind(lib, "GetBattleData", [_c_int],          _c_str)

#: char* Select(const char* select_json)
Select:         ctypes.CFUNCTYPE = _bind(lib, "Select",        [_c_str],          _c_str)

#: char* VisualizeData()
VisualizeData:  ctypes.CFUNCTYPE = _bind(lib, "VisualizeData", [],                _c_str)

#: char* SearchBegin(const char* input_json)
SearchBegin:    ctypes.CFUNCTYPE = _bind(lib, "SearchBegin",   [_c_str],          _c_str)

#: char* SearchStep(const char* step_json)
SearchStep:     ctypes.CFUNCTYPE = _bind(lib, "SearchStep",    [_c_str],          _c_str)

#: void SearchEnd()
SearchEnd:      ctypes.CFUNCTYPE = _bind(lib, "SearchEnd",     [],                _c_void)

#: void SearchRelease(const char* release_json)
SearchRelease:  ctypes.CFUNCTYPE = _bind(lib, "SearchRelease", [_c_str],          _c_void)

#: char* AllCard()
AllCard:        ctypes.CFUNCTYPE = _bind(lib, "AllCard",       [],                _c_str)

#: char* AllAttack()
AllAttack:      ctypes.CFUNCTYPE = _bind(lib, "AllAttack",     [],                _c_str)

# One-time engine initialisation — must happen before any BattleStart call.
GameInitialize()


# ---------------------------------------------------------------------------
# Helper: decode a c_char_p response to a Python str / dict
# ---------------------------------------------------------------------------

def _decode(raw: bytes | None) -> str | None:
    if raw is None:
        return None
    return raw.decode("utf-8")


def _decode_json(raw: bytes | None) -> dict | list | None:
    text = _decode(raw)
    if not text:
        return None
    return json.loads(text)


# ---------------------------------------------------------------------------
# StartData
# ---------------------------------------------------------------------------

class StartData:
    """
    Result object returned by BattleStart / Battle.start().

    Attributes
    ----------
    error : str | None
        None on success; human-readable error message on failure.
    player0 : int
        Engine-assigned index for the first deck (almost always 0).
    player1 : int
        Engine-assigned index for the second deck (almost always 1).
    """
    __slots__ = ("error", "player0", "player1")

    def __init__(
        self,
        error:   str | None = None,
        player0: int        = 0,
        player1: int        = 1,
    ) -> None:
        self.error   = error
        self.player0 = player0
        self.player1 = player1

    @classmethod
    def _from_response(cls, response: dict) -> "StartData":
        """Build from the parsed BattleStart JSON response."""
        return cls(
            error   = response.get("error"),
            player0 = response.get("player0", 0),
            player1 = response.get("player1", 1),
        )

    @property
    def ok(self) -> bool:
        """True when no error occurred."""
        return self.error is None

    def __repr__(self) -> str:
        if self.error:
            return f"StartData(error={self.error!r})"
        return f"StartData(player0={self.player0}, player1={self.player1})"


# ---------------------------------------------------------------------------
# SerialData
# ---------------------------------------------------------------------------

class SerialData:
    """
    Maps the serial numbers of in-play cards to their card-type IDs.

    The engine assigns a unique ``serial`` to every card instance at the
    start of a battle (two copies of the same card get different serials).
    SerialData lets you resolve serial → card ID → CardData metadata.

    Attributes
    ----------
    mapping : dict[int, int]
        serial → cardId
    """
    __slots__ = ("mapping",)

    def __init__(self, mapping: dict[int, int]) -> None:
        self.mapping = mapping

    @classmethod
    def from_battle(cls) -> "SerialData":
        """
        Fetch the current serial→cardId mapping from the engine.

        Calls GetBattleData for both players and merges the serial tables.
        """
        mapping: dict[int, int] = {}
        for player_index in (0, 1):
            raw      = GetBattleData(player_index)
            data     = _decode_json(raw) or {}
            serials  = data.get("serials", {})
            # serials may come as {"<serial>": cardId, ...} with string keys
            for k, v in serials.items():
                mapping[int(k)] = int(v)
        return cls(mapping)

    def card_id(self, serial: int) -> int | None:
        """Return the cardId for a given serial, or None if unknown."""
        return self.mapping.get(serial)

    def __len__(self) -> int:
        return len(self.mapping)

    def __repr__(self) -> str:
        return f"SerialData({len(self.mapping)} cards)"


# ---------------------------------------------------------------------------
# Battle
# ---------------------------------------------------------------------------

class Battle:
    """
    Stateful wrapper around one battle session.

    The cabt engine only supports a single concurrent battle per process
    (global C++ state).  Battle tracks whether a session is active and
    prevents double-starts / double-finishes.

    Usage
    -----
    >>> b = Battle(deck0, deck1)
    >>> obs_raw = b.start()           # starts the battle, returns JSON bytes
    >>> obs_raw = b.select([0])       # advances one step
    >>> b.finish()                    # frees C++ resources
    """

    __slots__ = ("_deck0", "_deck1", "_active", "_player0", "_player1")

    def __init__(self, deck0: list[int], deck1: list[int]) -> None:
        if len(deck0) != 60:
            raise ValueError(f"deck0 must have exactly 60 cards, got {len(deck0)}")
        if len(deck1) != 60:
            raise ValueError(f"deck1 must have exactly 60 cards, got {len(deck1)}")
        self._deck0:   list[int] = deck0
        self._deck1:   list[int] = deck1
        self._active:  bool      = False
        self._player0: int       = 0
        self._player1: int       = 1

    # ------------------------------------------------------------------
    # Core battle lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bytes | None:
        """
        Start the battle.

        Returns the raw JSON bytes of the initial observation.
        Raises RuntimeError if a battle is already active.
        """
        if self._active:
            raise RuntimeError("Battle already started; call finish() first.")

        deck0_json = json.dumps(self._deck0).encode("utf-8")
        deck1_json = json.dumps(self._deck1).encode("utf-8")

        raw = BattleStart(deck0_json, deck1_json)

        # Parse just enough to pull out player assignments and check for errors
        response = _decode_json(raw) or {}
        if response.get("error"):
            return raw  # caller (game.py) will surface the error

        self._player0 = response.get("player0", 0)
        self._player1 = response.get("player1", 1)
        self._active  = True
        return raw

    def agent_start(self, player_index: int) -> bytes | None:
        """
        Retrieve the initial observation from the perspective of one player.

        Called after start() when you need per-player views (e.g. in
        self-play where each agent runs in the same process).
        """
        self._require_active()
        return AgentStart(player_index)

    def select(self, select_list: list[int]) -> bytes | None:
        """
        Apply a selection and return the next observation as raw JSON bytes.

        Parameters
        ----------
        select_list : list[int]
            Indices into the current option list.  Pass [] for auto phases.
        """
        self._require_active()
        payload = json.dumps({"select": select_list}).encode("utf-8")
        return Select(payload)

    def finish(self) -> None:
        """
        End the battle and free all C++ resources.

        Safe to call even if the battle ended normally (result set).
        Idempotent — calling twice is a no-op.
        """
        if self._active:
            BattleFinish()
            self._active = False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_battle_data(self, player_index: int) -> bytes | None:
        """
        Return raw game state JSON for ``player_index``.

        Includes card serials, hidden information (hand), and deck counts
        for the requested player.  The other player's hand is omitted.
        """
        self._require_active()
        return GetBattleData(player_index)

    def serial_data(self) -> "SerialData":
        """Return a SerialData snapshot of the current serial→cardId mapping."""
        self._require_active()
        return SerialData.from_battle()

    def visualize(self) -> bytes | None:
        """Return a human-readable board state string (for debugging)."""
        self._require_active()
        return VisualizeData()

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Battle":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.finish()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("No active battle — call start() first.")

    @property
    def active(self) -> bool:
        """True while a battle session is running."""
        return self._active

    @property
    def player0(self) -> int:
        """Engine-assigned index for the first deck."""
        return self._player0

    @property
    def player1(self) -> int:
        """Engine-assigned index for the second deck."""
        return self._player1

    def __repr__(self) -> str:
        status = "active" if self._active else "idle"
        return f"Battle({status}, player0={self._player0}, player1={self._player1})"