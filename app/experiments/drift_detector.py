import asyncio
import logging
from app.core.config import settings
from app.observability.metrics import drift_alert_total

logger = logging.getLogger("inferroute")


class DriftDetector:

    def __init__(self, threshold: float = None, interval: int = None):
        self._threshold = threshold or settings.drift_threshold
        self._interval = interval or settings.drift_detection_interval
        self._champion_scores: list[float] = []
        self._challenger_scores: list[float] = []
        self._running = False
        self._task = None
        self._alerted = False

    def record_champion_score(self, score: float):
        self._champion_scores.append(score)

    def record_challenger_score(self, score: float):
        self._challenger_scores.append(score)

    async def check_drift(self) -> bool:
        if not self._champion_scores or not self._challenger_scores:
            return False

        champion_avg = sum(self._champion_scores) / len(self._champion_scores)
        challenger_avg = sum(self._challenger_scores) / len(self._challenger_scores)
        divergence = abs(champion_avg - challenger_avg)

        if divergence > self._threshold:
            logger.warning(
                "DRIFT DETECTED: champion_avg=%.3f challenger_avg=%.3f divergence=%.3f threshold=%.3f",
                champion_avg, challenger_avg, divergence, self._threshold,
            )
            drift_alert_total.inc()
            self._alerted = True
            return True

        return False

    async def _run_loop(self):
        while self._running:
            await asyncio.sleep(self._interval)
            await self.check_drift()

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Drift detector started (interval=%ds, threshold=%.2f)", self._interval, self._threshold)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @property
    def alerted(self) -> bool:
        return self._alerted

    def reset(self):
        self._champion_scores.clear()
        self._challenger_scores.clear()
        self._alerted = False
