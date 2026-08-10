import locale
import os

from gazetype.runtime import configure_numeric_locale


def test_configure_numeric_locale_forces_c_decimal_parsing() -> None:
    original_all = os.environ.get("LC_ALL")
    original_lang = os.environ.get("LANG")
    original_env = os.environ.get("LC_NUMERIC")
    original_locale = locale.setlocale(locale.LC_ALL)
    try:
        locale.setlocale(locale.LC_ALL, "tr_TR.utf8")
        configure_numeric_locale()
        assert locale.localeconv()["decimal_point"] == "."
        assert os.environ["LC_ALL"] == "C"
        assert os.environ["LANG"] == "C"
        assert os.environ["LC_NUMERIC"] == "C"
    finally:
        if original_all is None:
            os.environ.pop("LC_ALL", None)
        else:
            os.environ["LC_ALL"] = original_all
        if original_lang is None:
            os.environ.pop("LANG", None)
        else:
            os.environ["LANG"] = original_lang
        if original_env is None:
            os.environ.pop("LC_NUMERIC", None)
        else:
            os.environ["LC_NUMERIC"] = original_env
        locale.setlocale(locale.LC_NUMERIC, original_locale)
