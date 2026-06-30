from __future__ import annotations

from dataclasses import dataclass

from gazetype.models import KeyboardLayout


BACKSPACE = "BACKSPACE"
ENTER = "ENTER"
SPACE = "SPACE"


@dataclass(frozen=True, slots=True)
class Key:
    id: str
    label: str
    output: str
    x: float
    y: float
    width: float
    height: float

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height


_LAYOUT_ROWS: dict[KeyboardLayout, tuple[tuple[tuple[str, str, float], ...], ...]] = {
    KeyboardLayout.TURKISH_Q: (
        tuple((char, char, 1.0) for char in "1234567890") + ((BACKSPACE, "⌫", 1.45),),
        tuple((char, char, 1.0) for char in "qwertyuıopğü"),
        tuple((char, char, 1.0) for char in "asdfghjklşi") + ((ENTER, "Enter", 1.55),),
        tuple((char, char, 1.0) for char in "zxcvbnmöç"),
        ((SPACE, "Boşluk", 6.0),),
    ),
    KeyboardLayout.ENGLISH_QWERTY: (
        tuple((char, char, 1.0) for char in "1234567890") + ((BACKSPACE, "⌫", 1.45),),
        tuple((char, char, 1.0) for char in "qwertyuiop"),
        tuple((char, char, 1.0) for char in "asdfghjkl") + ((ENTER, "Enter", 1.55),),
        tuple((char, char, 1.0) for char in "zxcvbnm"),
        ((SPACE, "Space", 6.0),),
    ),
}


class KeyboardGeometry:
    """Normalized screen-space keyboard geometry."""

    def __init__(self, layout: KeyboardLayout, top: float = 0.57, bottom: float = 0.98):
        self.layout = layout
        self.top = top
        self.bottom = bottom
        self.keys = self._build_keys()

    def _build_keys(self) -> tuple[Key, ...]:
        rows = _LAYOUT_ROWS[self.layout]
        row_gap = 0.008
        key_gap = 0.006
        side_margin = 0.018
        row_height = (self.bottom - self.top - row_gap * (len(rows) - 1)) / len(rows)
        keys: list[Key] = []
        for row_index, row in enumerate(rows):
            available = 1.0 - 2 * side_margin - key_gap * (len(row) - 1)
            total_weight = sum(item[2] for item in row)
            unit = available / total_weight
            row_width = sum(item[2] * unit for item in row) + key_gap * (len(row) - 1)
            x = (1.0 - row_width) / 2
            y = self.top + row_index * (row_height + row_gap)
            for key_id, label, weight in row:
                width = unit * weight
                output = " " if key_id == SPACE else key_id
                keys.append(Key(key_id, label, output, x, y, width, row_height))
                x += width + key_gap
        return tuple(keys)

    def hit_test(self, x: float, y: float) -> Key | None:
        for key in self.keys:
            if key.contains(x, y):
                return key
        return None

    def by_id(self, key_id: str) -> Key:
        return next(key for key in self.keys if key.id == key_id)

