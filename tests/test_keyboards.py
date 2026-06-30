from gazetype.keyboards import BACKSPACE, ENTER, SPACE, KeyboardGeometry
from gazetype.models import KeyboardLayout


def test_turkish_layout_contains_expected_keys() -> None:
    keyboard = KeyboardGeometry(KeyboardLayout.TURKISH_Q)
    ids = {key.id for key in keyboard.keys}
    assert {"ğ", "ü", "ş", "ı", "ö", "ç", BACKSPACE, ENTER, SPACE} <= ids


def test_hit_test_returns_key_center() -> None:
    keyboard = KeyboardGeometry(KeyboardLayout.ENGLISH_QWERTY)
    key = keyboard.by_id("g")
    assert keyboard.hit_test(key.x + key.width / 2, key.y + key.height / 2) == key
    assert keyboard.hit_test(0.5, 0.995) is None


def test_default_keyboard_uses_nearly_full_screen() -> None:
    keyboard = KeyboardGeometry(KeyboardLayout.TURKISH_Q)
    assert keyboard.top <= 0.03
    assert keyboard.bottom >= 0.97
    assert max(key.y + key.height for key in keyboard.keys) >= 0.97
