from __future__ import annotations

from gazetype.input import LinuxInputSender, UnsupportedInputSender
from gazetype.keyboards import BACKSPACE, ENTER, SPACE


def test_linux_input_sender_maps_special_keys() -> None:
    sender = LinuxInputSender()

    assert sender._command(BACKSPACE, BACKSPACE)[-1] == "BackSpace"
    assert sender._command(ENTER, ENTER)[-1] == "Return"
    assert sender._command(SPACE, " ")[-1] == "space"


def test_linux_input_sender_types_text() -> None:
    sender = LinuxInputSender()

    assert sender._command("ğ", "ğ")[-2:] == ["--clearmodifiers", "ğ"]


def test_unsupported_input_sender_reports_failure() -> None:
    assert not UnsupportedInputSender().send("a", "a")
