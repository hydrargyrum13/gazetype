from pathlib import Path

from gazetype.settings import AppSettings, SettingsStore


def test_oversized_settings_file_falls_back(tmp_path: Path):
    path = tmp_path / "settings.json"
    path.write_text("{" + (" " * (1024 * 1024 + 1)) + "}", encoding="utf-8")
    assert SettingsStore(path).load() == AppSettings()


def test_non_object_settings_fall_back(tmp_path: Path):
    path = tmp_path / "settings.json"
    path.write_text("[]", encoding="utf-8")
    assert SettingsStore(path).load() == AppSettings()
