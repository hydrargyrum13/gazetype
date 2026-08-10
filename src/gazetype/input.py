from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Protocol

from gazetype.keyboards import BACKSPACE, ENTER, SPACE


class InputSender(Protocol):
    def send(self, key_id: str, output: str) -> bool:
        ...


class LinuxInputSender:
    def __init__(self) -> None:
        self._xdotool = shutil.which("xdotool")
        self._session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()

    def send(self, key_id: str, output: str) -> bool:
        if self._xdotool is None or self._session_type == "wayland":
            return False
        command = self._command(key_id, output)
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            return False
        return True

    def _command(self, key_id: str, output: str) -> list[str]:
        key_names = {
            BACKSPACE: "BackSpace",
            ENTER: "Return",
            SPACE: "space",
        }
        if key_id in key_names:
            return [self._xdotool or "xdotool", "key", key_names[key_id]]
        return [self._xdotool or "xdotool", "type", "--clearmodifiers", output]


class UnsupportedInputSender:
    def send(self, key_id: str, output: str) -> bool:
        return False


def create_input_sender() -> InputSender:
    if sys.platform == "win32":
        from gazetype.input_windows import WindowsInputSender

        return WindowsInputSender()
    if sys.platform.startswith("linux"):
        return LinuxInputSender()
    return UnsupportedInputSender()
