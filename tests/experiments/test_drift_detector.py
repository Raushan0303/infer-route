import pytest
import asyncio
from app.experiments.drift_detector import DriftDetector


@pytest.mark.asyncio
async def test_no_drift_when_scores_similar():
    detector = DriftDetector(threshold=0.1, interval=300)
    for _ in range(10):
        detector.record_champion_score(0.9)
        detector.record_challenger_score(0.88)

    drifted = await detector.check_drift()
    assert drifted is False
    assert detector.alerted is False


@pytest.mark.asyncio
async def test_drift_detected_when_divergence_exceeds_threshold():
    detector = DriftDetector(threshold=0.1, interval=300)
    for _ in range(10):
        detector.record_champion_score(0.9)
        detector.record_challenger_score(0.5)  # 0.4 divergence > 0.1 threshold

    drifted = await detector.check_drift()
    assert drifted is True
    assert detector.alerted is True


@pytest.mark.asyncio
async def test_no_drift_with_empty_scores():
    detector = DriftDetector(threshold=0.1, interval=300)
    drifted = await detector.check_drift()
    assert drifted is False


@pytest.mark.asyncio
async def test_drift_detector_start_stop():
    detector = DriftDetector(threshold=0.1, interval=1)
    await detector.start()
    assert detector._running is True
    await detector.stop()
    assert detector._running is False


@pytest.mark.asyncio
async def test_drift_detector_reset():
    detector = DriftDetector(threshold=0.1, interval=300)
    detector.record_champion_score(0.9)
    detector.record_challenger_score(0.5)
    await detector.check_drift()
    assert detector.alerted is True

    detector.reset()
    assert detector.alerted is False
    assert len(detector._champion_scores) == 0
