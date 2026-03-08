#!/usr/bin/env python3
"""
Empirical Bayes shrinkage estimator for consensus scores.

Reads per-term consensus data, adjusts for rater bias, penalizes small
sample sizes, and accounts for inter-rater disagreement.  Writes a single
JSON file consumed by the stats page and term modals.

Usage:
    python bot/bayes_scores.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONSENSUS_DIR = REPO_ROOT / "bot" / "consensus-data"
OUTPUT_PATH = REPO_ROOT / "docs" / "api" / "v1" / "bayes-scores.json"


def collect_ratings():
    """Extract flat list of (slug, name, model, recognition, timestamp) from consensus files."""
    ratings = []
    for path in sorted(CONSENSUS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        slug = data.get("slug", path.stem)
        name = data.get("name", slug)
        for rnd in data.get("rounds", []):
            for model, info in rnd.get("ratings", {}).items():
                score = info.get("recognition")
                if score is not None:
                    ratings.append({
                        "slug": slug,
                        "name": name,
                        "model": model,
                        "recognition": score,
                        "timestamp": info.get("timestamp", ""),
                    })
    return ratings


def compute_bayes_scores(ratings):
    """Run the 10-step Empirical Bayes algorithm.

    Returns (global_stats, rater_biases, term_results).
    """
    if not ratings:
        return {}, {}, []

    # Step 1: Grand mean
    all_scores = [r["recognition"] for r in ratings]
    grand_mean = sum(all_scores) / len(all_scores)

    # Step 2: Per-rater bias
    rater_totals = {}
    rater_counts = {}
    for r in ratings:
        m = r["model"]
        rater_totals[m] = rater_totals.get(m, 0) + r["recognition"]
        rater_counts[m] = rater_counts.get(m, 0) + 1
    rater_biases = {
        m: rater_totals[m] / rater_counts[m] - grand_mean for m in rater_totals
    }

    # Step 3: Bias-adjusted scores
    adjusted = []
    for r in ratings:
        adjusted.append({
            **r,
            "adjusted": r["recognition"] - rater_biases[r["model"]],
        })

    # Step 4: Per-term adjusted mean + count
    term_sums = {}
    term_counts = {}
    term_names = {}
    term_scores = {}  # slug -> list of adjusted scores
    for a in adjusted:
        s = a["slug"]
        term_sums[s] = term_sums.get(s, 0) + a["adjusted"]
        term_counts[s] = term_counts.get(s, 0) + 1
        term_names[s] = a["name"]
        term_scores.setdefault(s, []).append(a["adjusted"])

    term_means = {s: term_sums[s] / term_counts[s] for s in term_sums}

    # Step 5: Variance components
    # tau^2 = between-term variance of adjusted means
    mean_of_means = sum(term_means.values()) / len(term_means)
    tau_sq = sum((m - mean_of_means) ** 2 for m in term_means.values()) / len(term_means)

    # sigma^2 = pooled within-term variance
    within_vars = []
    for s, scores in term_scores.items():
        if len(scores) > 1:
            m = term_means[s]
            v = sum((x - m) ** 2 for x in scores) / (len(scores) - 1)
            within_vars.append((len(scores) - 1, v))

    if within_vars:
        total_df = sum(df for df, _ in within_vars)
        sigma_sq = sum(df * v for df, v in within_vars) / total_df
    else:
        sigma_sq = 1.0  # fallback

    # Step 6-10: Per-term results
    results = []
    for slug in sorted(term_sums):
        n = term_counts[slug]
        tm = term_means[slug]

        # Step 6: Shrinkage
        if tau_sq == 0:
            shrinkage = 0.0
        else:
            shrinkage = tau_sq / (tau_sq + sigma_sq / n)

        # Step 7: Shrunk estimate
        shrunk = grand_mean + shrinkage * (tm - grand_mean)

        # Step 8: Base confidence (map 1-7 to 0-1)
        base_confidence = (shrunk - 1) / 6

        # Step 9: Agreement
        scores = term_scores[slug]
        if len(scores) > 1:
            var = sum((x - tm) ** 2 for x in scores) / (len(scores) - 1)
        else:
            var = sigma_sq  # fallback for n=1
        agreement = max(0, 1 - var / 9)

        # Step 10: Final consensus score
        credibility = 1 - 1 / (1 + n)
        consensus_score = (0.7 * base_confidence + 0.3 * agreement) * credibility

        results.append({
            "slug": slug,
            "name": term_names[slug],
            "consensus_score": round(consensus_score, 4),
            "shrunk_estimate": round(shrunk, 4),
            "raw_mean": round(tm, 4),
            "shrinkage_factor": round(shrinkage, 4),
            "agreement": round(agreement, 4),
            "n_ratings": n,
            "credibility": round(credibility, 4),
        })

    # Sort by consensus_score descending
    results.sort(key=lambda t: t["consensus_score"], reverse=True)

    global_stats = {
        "grand_mean": round(grand_mean, 4),
        "tau_squared": round(tau_sq, 4),
        "sigma_squared": round(sigma_sq, 4),
        "total_ratings": len(ratings),
        "total_terms": len(term_sums),
    }

    return global_stats, rater_biases, results


def main():
    ratings = collect_ratings()
    if not ratings:
        print("No ratings found.")
        return

    global_stats, rater_biases, terms = compute_bayes_scores(ratings)

    output = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "algorithm": "empirical_bayes_v1",
        "global_stats": global_stats,
        "rater_biases": {k: round(v, 4) for k, v in sorted(rater_biases.items())},
        "terms": terms,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(terms)} term scores to {OUTPUT_PATH}")
    print(f"  Grand mean: {global_stats['grand_mean']}")
    print(f"  Total ratings: {global_stats['total_ratings']}")


if __name__ == "__main__":
    main()
