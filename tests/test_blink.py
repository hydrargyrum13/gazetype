from gazetype.blink import DeliberateBlinkDetector


def test_deliberate_blink_triggers_only_when_armed() -> None:
    detector = DeliberateBlinkDetector()
    assert not detector.update(0, 0.8, 0.8, True)
    assert detector.update(300, 0.1, 0.1, True)

    assert not detector.update(1200, 0.8, 0.8, False)
    assert not detector.update(1500, 0.1, 0.1, False)


def test_short_natural_blink_is_ignored() -> None:
    detector = DeliberateBlinkDetector()
    detector.update(0, 0.9, 0.9, True)
    assert not detector.update(120, 0.1, 0.1, True)

