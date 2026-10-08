"""Estimate what complexity routing saves, using the REAL classifier + strategy.

Runs every labeled prompt in benchmarks/data/routing_prompts.json through
ComplexityClassifier + IntelligenceAware (no LLM classifier, heuristic only)
with three providers registered at their list prices:

    cheap    gpt-4o-mini              $0.15 / $0.60 per 1M
    standard llama-3.3-70b (Groq)     $0.59 / $0.79 per 1M
    premium  gpt-4o                   $2.50 / $10.00 per 1M

and prices each request with app.routing.cost.compute_call_cost.

This is an ESTIMATE, not billed spend:
  * input tokens ≈ characters / 4
  * output tokens are an assumed length per true label (same for every model)
  * answer quality is NOT measured

Usage: python -m benchmarks.routing_cost
Writes benchmarks/results/routing_cost.json
"""
import asyncio
import json
import pathlib
from collections import Counter

from app.gateway.models import InferRouteMessage, InferRouteRequest
from app.routing.complexity_classifier import ComplexityClassifier
from app.routing.cost import compute_call_cost
from app.routing.pricing import prices_for
from app.routing.provider_registry import ProviderRegistry
from app.routing.strategies import IntelligenceAware

HERE = pathlib.Path(__file__).parent
ASSUMED_OUTPUT_TOKENS = {"simple": 60, "medium": 350, "complex": 900}
PROVIDERS = [  # provider_id, model, tier
    ("openai-mini", "gpt-4o-mini", "cheap"),
    ("groq", "llama-3.3-70b-versatile", "standard"),
    ("openai-4o", "gpt-4o", "premium"),
]


def cost_of(model, in_tok, out_tok):
    cin, cout = prices_for(model)
    return compute_call_cost(cin, cout, in_tok, out_tok)


async def main():
    prompts = json.loads((HERE / "data" / "routing_prompts.json").read_text())
    strategy = IntelligenceAware(classifier=ComplexityClassifier())
    registry = ProviderRegistry(strategy=strategy)
    model_of = {}
    for pid, model, tier in PROVIDERS:
        cin, cout = prices_for(model)
        registry.register(pid, adapter=None, base_url="", tier=tier,
                          cost_per_1m_input=cin, cost_per_1m_output=cout)
        model_of[pid] = model

    rows, routed_to = [], Counter()
    totals = {"routed": 0.0, "all_gpt_4o": 0.0, "all_gpt_4o_mini": 0.0, "all_llama_70b": 0.0}
    confusion = Counter()
    for p in prompts:
        req = InferRouteRequest(model="auto", messages=[InferRouteMessage(role="user", content=p["prompt"])])
        predicted, _ = await strategy.classify_complexity(req)
        chosen = await registry.select(request=req)
        in_tok = max(1, len(p["prompt"]) // 4)
        out_tok = ASSUMED_OUTPUT_TOKENS[p["label"]]
        routed_cost = cost_of(model_of[chosen.provider_id], in_tok, out_tok)
        totals["routed"] += routed_cost
        totals["all_gpt_4o"] += cost_of("gpt-4o", in_tok, out_tok)
        totals["all_gpt_4o_mini"] += cost_of("gpt-4o-mini", in_tok, out_tok)
        totals["all_llama_70b"] += cost_of("llama-3.3-70b-versatile", in_tok, out_tok)
        routed_to[model_of[chosen.provider_id]] += 1
        confusion[(p["label"], predicted)] += 1
        rows.append({"label": p["label"], "predicted": predicted,
                     "model": model_of[chosen.provider_id], "cost_usd": routed_cost})

    accuracy = sum(v for (t, pr), v in confusion.items() if t == pr) / len(prompts)
    result = {
        "prompts": len(prompts),
        "assumptions": {"input_tokens": "chars/4", "output_tokens_by_label": ASSUMED_OUTPUT_TOKENS,
                        "classifier": "heuristic only (no LLM tier)", "quality": "not measured"},
        "classifier_accuracy_vs_labels": round(accuracy, 3),
        "confusion_true_to_predicted": {f"{t}->{pr}": v for (t, pr), v in sorted(confusion.items())},
        "routed_to": dict(routed_to),
        "total_cost_usd": {k: round(v, 6) for k, v in totals.items()},
        "routed_vs_all_gpt_4o_saving_pct": round(100 * (1 - totals["routed"] / totals["all_gpt_4o"]), 1),
        "routed_vs_all_gpt_4o_mini_saving_pct": round(100 * (1 - totals["routed"] / totals["all_gpt_4o_mini"]), 1),
        "rows": rows,
    }
    out = HERE / "results" / "routing_cost.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
