import ctypes

from gazetype.input_windows import INPUT, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, key_events
from gazetype.keyboards import BACKSPACE


def test_unicode_character_uses_unicode_scan_code() -> None:
    down, up = key_events("ğ", "ğ")
    assert down[0] == 0
    assert down[2] == KEYEVENTF_UNICODE
    assert up[2] == KEYEVENTF_UNICODE | KEYEVENTF_KEYUP


def test_special_key_uses_virtual_key() -> None:
    down, up = key_events(BACKSPACE, BACKSPACE)
    assert down[0] != 0
    assert up[2] == KEYEVENTF_KEYUP


def test_input_structure_matches_win64_abi() -> None:
    assert ctypes.sizeof(INPUT) == 40
