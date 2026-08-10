from __future__ import annotations

import locale
import os


def configure_numeric_locale() -> None:
    os.environ["LC_ALL"] = "C"
    os.environ["LANG"] = "C"
    os.environ["LC_NUMERIC"] = "C"
    try:
        locale.setlocale(locale.LC_ALL, "C")
    except locale.Error:
        try:
            locale.setlocale(locale.LC_NUMERIC, "C")
        except locale.Error:
            pass
