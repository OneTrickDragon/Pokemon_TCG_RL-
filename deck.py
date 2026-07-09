"""Utilities for reading and validating PTCGABC deck files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


DECK_SIZE = 60


def load_deck_csv(path: str | Path, allowed_ids: Iterable[int] | None = None) -> list[int]:
    """
    Load a Kaggle-format deck file.

    The expected format is one integer card ID per non-empty line. The returned
    deck is validated to contain exactly 60 card IDs.
    """
    deck_path = Path(path)
    card_ids: list[int] = []
    for line_number, raw_line in enumerate(deck_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            card_ids.append(int(line))
        except ValueError as exc:
            raise ValueError(f"{deck_path}:{line_number}: expected integer card ID, got {line!r}") from exc

    validate_deck(card_ids, allowed_ids=allowed_ids)
    return card_ids


def write_deck_csv(path: str | Path, card_ids: Iterable[int]) -> None:
    """Write a validated 60-card deck as one card ID per line."""
    deck = list(card_ids)
    validate_deck(deck)
    text = "".join(f"{card_id}\n" for card_id in deck)
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def validate_deck(card_ids: Iterable[int], allowed_ids: Iterable[int] | None = None) -> None:
    """Raise `ValueError` if `card_ids` is not a valid 60-card ID list."""
    deck = list(card_ids)
    if len(deck) != DECK_SIZE:
        raise ValueError(f"deck must contain exactly {DECK_SIZE} cards, got {len(deck)}")
    if not all(isinstance(card_id, int) for card_id in deck):
        raise ValueError("deck must contain only integer card IDs")
    if any(card_id < 0 for card_id in deck):
        raise ValueError("deck must not contain negative card IDs")

    if allowed_ids is not None:
        allowed = set(allowed_ids)
        illegal = sorted({card_id for card_id in deck if card_id not in allowed})
        if illegal:
            raise ValueError(f"deck contains card IDs outside the allowed pool: {illegal}")
