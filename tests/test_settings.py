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


def test_gaze_average_count_is_persisted_and_bounded() -> None:
    settings = AppSettings(gaze_average_count=12)
    assert AppSettings.from_dict(settings.to_dict()).gaze_average_count == 12
    assert AppSettings(gaze_average_count=100).gaze_average_count == 30
    assert AppSettings(gaze_average_count=0).gaze_average_count == 1


def test_keyboard_calibration_mode_is_persisted() -> None:
    settings = AppSettings(calibration_mode="keyboard")
    assert AppSettings.from_dict(settings.to_dict()).calibration_mode == "keyboard"
    assert AppSettings(calibration_mode="unknown").calibration_mode == "grid"


def test_outdated_calibration_does_not_reset_other_settings() -> None:
    settings = AppSettings.from_dict({
        "camera_index": 2,
        "gaze_average_count": 9,
        "calibration": {"model_version": 2, "coefficients": []},
    })
    assert settings.camera_index == 2
    assert settings.gaze_average_count == 9
    assert settings.calibration is None


def test_experimental_sensitivity_settings_are_bounded() -> None:
    settings = AppSettings(
        horizontal_gain_percent=300,
        vertical_gain_percent=20,
        vertical_offset_percent=40,
        head_compensation_percent=200,
        head_motion_threshold_percent=10,
    )
    assert settings.horizontal_gain_percent == 200
    assert settings.vertical_gain_percent == 50
    assert settings.vertical_offset_percent == 25
    assert settings.head_compensation_percent == 150
    assert settings.head_motion_threshold_percent == 40
