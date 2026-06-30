from gazetype.models import KeyboardLayout, Sensitivity
from gazetype.settings import AppSettings, SettingsStore


def test_settings_round_trip(tmp_path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    expected = AppSettings(
        camera_index=2,
        screen_name="Monitor 2",
        screen_geometry="1920,0,1920,1080",
        layout=KeyboardLayout.ENGLISH_QWERTY,
        sensitivity=Sensitivity.BALANCED,
    )
    store.save(expected)
    actual = store.load()
    assert actual.camera_index == 2
    assert actual.layout == KeyboardLayout.ENGLISH_QWERTY
    assert actual.sensitivity == Sensitivity.BALANCED


def test_settings_normalizes_qt_string_enum_values() -> None:
    settings = AppSettings(layout="tr_q", sensitivity="fast")  # type: ignore[arg-type]
    serialized = settings.to_dict()
    assert settings.layout is KeyboardLayout.TURKISH_Q
    assert settings.sensitivity is Sensitivity.FAST
    assert serialized["layout"] == "tr_q"
    assert serialized["sensitivity"] == "fast"
