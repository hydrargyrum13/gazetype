from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

from gazetype.keyboards import BACKSPACE, ENTER, SPACE


INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
VK_BACK = 0x08
VK_RETURN = 0x0D
VK_SPACE = 0x20
ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD))


class _INPUTUNION(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT))


class INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = (("type", wintypes.DWORD), ("union", _INPUTUNION))


def key_events(key_id: str, output: str) -> list[tuple[int, int, int]]:
    virtual_keys = {BACKSPACE: VK_BACK, ENTER: VK_RETURN, SPACE: VK_SPACE}
    if key_id in virtual_keys:
        vk = virtual_keys[key_id]
        return [(vk, 0, 0), (vk, 0, KEYEVENTF_KEYUP)]
    events: list[tuple[int, int, int]] = []
    for character in output:
        encoded = character.encode("utf-16-le")
        for index in range(0, len(encoded), 2):
            code_unit = int.from_bytes(encoded[index:index + 2], "little")
            events.append((0, code_unit, KEYEVENTF_UNICODE))
            events.append((0, code_unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
    return events


class WindowsInputSender:
    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("WindowsInputSender is only available on Windows")
        self._send_input = ctypes.windll.user32.SendInput

    def send(self, key_id: str, output: str) -> bool:
        encoded = key_events(key_id, output)
        inputs = (INPUT * len(encoded))()
        for index, (virtual_key, scan_code, flags) in enumerate(encoded):
            inputs[index].type = INPUT_KEYBOARD
            inputs[index].ki = KEYBDINPUT(virtual_key, scan_code, flags, 0, 0)
        sent = self._send_input(len(inputs), inputs, ctypes.sizeof(INPUT))
        return sent == len(inputs)
