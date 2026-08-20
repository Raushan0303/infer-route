import logging
from dataclasses import dataclass, field
from app.observability.metrics import feedback_total
from app.core.database import db

logger = logging.getLogger("inferroute")


@dataclass
class FeedbackEntry:
    trace_id: str
    thumbs: str
    comment: str = ""
    model: str = ""
    provider: str = ""
    variant: str = ""


class FeedbackStore:

    def __init__(self):
        self._entries: list[FeedbackEntry] = []
        self._db = db

    async def record(
        self,
        trace_id: str,
        thumbs: str,
        comment: str = "",
        model: str = "",
        provider: str = "",
        variant: str = "",
    ) -> None:
        entry = FeedbackEntry(
            trace_id=trace_id,
            thumbs=thumbs,
            comment=comment,
            model=model,
            provider=provider,
            variant=variant,
        )
        self._entries.append(entry)
        feedback_total.labels(thumbs=thumbs).inc()
        logger.info("Feedback recorded: trace_id=%s thumbs=%s", trace_id, thumbs)

        try:
            pool = await self._db.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO feedback (trace_id, thumbs, comment, model, provider, variant)
                    VALUES ($1, $2, $3, $4, $5, $6)""",
                    trace_id, thumbs, comment, model, provider, variant,
                )
        except Exception as e:
            logger.warning("Failed to persist feedback to Postgres (in-memory only): %s", e)

    async def get_stats(self, model: str = None) -> dict:
        try:
            pool = await self._db.get_pool()
            async with pool.acquire() as conn:
                if model:
                    row = await conn.fetchrow(
                        """SELECT
                            COUNT(*) as total,
                            COUNT(*) FILTER (WHERE thumbs = 'up') as thumbs_up,
                            COUNT(*) FILTER (WHERE thumbs = 'down') as thumbs_down
                        FROM feedback WHERE model = $1""",
                        model,
                    )
                else:
                    row = await conn.fetchrow(
                        """SELECT
                            COUNT(*) as total,
                            COUNT(*) FILTER (WHERE thumbs = 'up') as thumbs_up,
                            COUNT(*) FILTER (WHERE thumbs = 'down') as thumbs_down
                        FROM feedback"""
                    )

                if row is None or row["total"] == 0:
                    return {"total": 0, "thumbs_up": 0, "thumbs_down": 0, "up_rate": 0.0}

                return {
                    "total": row["total"],
                    "thumbs_up": row["thumbs_up"],
                    "thumbs_down": row["thumbs_down"],
                    "up_rate": row["thumbs_up"] / row["total"] if row["total"] > 0 else 0.0,
                }
        except Exception as e:
            logger.warning("Failed to query feedback from Postgres (using in-memory): %s", e)
            filtered = [e for e in self._entries if model is None or e.model == model]
            total = len(filtered)
            if total == 0:
                return {"total": 0, "thumbs_up": 0, "thumbs_down": 0, "up_rate": 0.0}
            up = sum(1 for e in filtered if e.thumbs == "up")
            down = sum(1 for e in filtered if e.thumbs == "down")
            return {
                "total": total,
                "thumbs_up": up,
                "thumbs_down": down,
                "up_rate": up / total,
            }

    def get_entries(self) -> list[FeedbackEntry]:
        return list(self._entries)
