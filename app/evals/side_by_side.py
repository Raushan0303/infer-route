import logging
from dataclasses import dataclass

logger = logging.getLogger("inferroute")


@dataclass
class EvalReport:
    variant_a_cost: float = 0.0
    variant_b_cost: float = 0.0
    variant_a_latency_p50: float = 0.0
    variant_b_latency_p50: float = 0.0
    variant_a_quality: float = 0.0
    variant_b_quality: float = 0.0
    sample_size_a: int = 0
    sample_size_b: int = 0
    cost_winner: str = ""
    latency_winner: str = ""
    quality_winner: str = ""
    overall_winner: str = ""


class SideBySideEval:

    async def compare(
        self,
        variant_a_results: list[dict],
        variant_b_results: list[dict],
    ) -> EvalReport:
        report = EvalReport()

        if variant_a_results:
            report.variant_a_cost = sum(r["cost"] for r in variant_a_results) / len(variant_a_results)
            latencies_a = sorted(r["latency_ms"] for r in variant_a_results)
            report.variant_a_latency_p50 = latencies_a[len(latencies_a) // 2]
            scores_a = [r.get("score") for r in variant_a_results if r.get("score") is not None]
            if scores_a:
                report.variant_a_quality = sum(scores_a) / len(scores_a)
            report.sample_size_a = len(variant_a_results)

        if variant_b_results:
            report.variant_b_cost = sum(r["cost"] for r in variant_b_results) / len(variant_b_results)
            latencies_b = sorted(r["latency_ms"] for r in variant_b_results)
            report.variant_b_latency_p50 = latencies_b[len(latencies_b) // 2]
            scores_b = [r.get("score") for r in variant_b_results if r.get("score") is not None]
            if scores_b:
                report.variant_b_quality = sum(scores_b) / len(scores_b)
            report.sample_size_b = len(variant_b_results)

        # Determine winners (lower cost/latency is better, higher quality is better)
        if report.variant_a_cost > 0 and report.variant_b_cost > 0:
            if report.variant_a_cost < report.variant_b_cost:
                report.cost_winner = "A"
            elif report.variant_b_cost < report.variant_a_cost:
                report.cost_winner = "B"

        if report.variant_a_latency_p50 > 0 and report.variant_b_latency_p50 > 0:
            if report.variant_a_latency_p50 < report.variant_b_latency_p50:
                report.latency_winner = "A"
            elif report.variant_b_latency_p50 < report.variant_a_latency_p50:
                report.latency_winner = "B"

        if report.variant_a_quality > 0 and report.variant_b_quality > 0:
            if report.variant_a_quality > report.variant_b_quality:
                report.quality_winner = "A"
            elif report.variant_b_quality > report.variant_a_quality:
                report.quality_winner = "B"

        # Overall winner: majority of dimensions
        wins_a = sum(1 for w in [report.cost_winner, report.latency_winner, report.quality_winner] if w == "A")
        wins_b = sum(1 for w in [report.cost_winner, report.latency_winner, report.quality_winner] if w == "B")
        if report.sample_size_a == 0 and report.sample_size_b == 0:
            report.overall_winner = ""
        elif wins_a > wins_b:
            report.overall_winner = "A"
        elif wins_b > wins_a:
            report.overall_winner = "B"
        else:
            report.overall_winner = "tie"

        logger.info(
            "Eval complete: winner=%s cost_a=%.2f cost_b=%.2f lat_a=%.1f lat_b=%.1f qual_a=%.2f qual_b=%.2f",
            report.overall_winner,
            report.variant_a_cost, report.variant_b_cost,
            report.variant_a_latency_p50, report.variant_b_latency_p50,
            report.variant_a_quality, report.variant_b_quality,
        )

        return report
