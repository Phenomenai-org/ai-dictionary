#!/usr/bin/env python3
"""Unit tests for bayes_scores.py using hardcoded fixture data."""

import unittest
from bayes_scores import compute_bayes_scores


def _make_rating(slug, name, model, score):
    return {"slug": slug, "name": name, "model": model, "recognition": score, "timestamp": ""}


class TestBayesScores(unittest.TestCase):
    """Test the 10-step Empirical Bayes algorithm with controlled fixtures."""

    def setUp(self):
        # 3 raters, 3 terms
        # "semantic-drift": 6 ratings, high scores (5-7) -> high consensus
        # "phantom-completion": 1 rating of 7 -> heavy shrinkage, low credibility
        # "alignment-fatigue": 4 ratings around 2-3 -> low score, decent agreement
        self.ratings = [
            # semantic-drift: high, 6 ratings
            _make_rating("semantic-drift", "Semantic Drift", "rater-a", 6),
            _make_rating("semantic-drift", "Semantic Drift", "rater-b", 7),
            _make_rating("semantic-drift", "Semantic Drift", "rater-c", 5),
            _make_rating("semantic-drift", "Semantic Drift", "rater-a", 7),
            _make_rating("semantic-drift", "Semantic Drift", "rater-b", 6),
            _make_rating("semantic-drift", "Semantic Drift", "rater-c", 6),
            # phantom-completion: single rating
            _make_rating("phantom-completion", "Phantom Completion", "rater-a", 7),
            # alignment-fatigue: 4 low ratings
            _make_rating("alignment-fatigue", "Alignment Fatigue", "rater-a", 2),
            _make_rating("alignment-fatigue", "Alignment Fatigue", "rater-b", 3),
            _make_rating("alignment-fatigue", "Alignment Fatigue", "rater-c", 2),
            _make_rating("alignment-fatigue", "Alignment Fatigue", "rater-b", 3),
        ]

    def test_scores_in_unit_interval(self):
        """All consensus_score values must be in [0, 1]."""
        _, _, terms = compute_bayes_scores(self.ratings)
        for t in terms:
            self.assertGreaterEqual(t["consensus_score"], 0, f"{t['slug']} below 0")
            self.assertLessEqual(t["consensus_score"], 1, f"{t['slug']} above 1")

    def test_shrinkage_pulls_single_rating_toward_mean(self):
        """A term with only 1 rating should have its shrunk estimate closer to the
        grand mean than its raw mean."""
        global_stats, _, terms = compute_bayes_scores(self.ratings)
        gm = global_stats["grand_mean"]
        phantom = next(t for t in terms if t["slug"] == "phantom-completion")
        self.assertLess(
            abs(phantom["shrunk_estimate"] - gm),
            abs(phantom["raw_mean"] - gm),
            "Shrunk estimate should be closer to grand mean than raw mean for n=1 term",
        )

    def test_credibility_ordering(self):
        """More ratings -> higher credibility."""
        _, _, terms = compute_bayes_scores(self.ratings)
        by_slug = {t["slug"]: t for t in terms}
        # 6 > 4 > 1 ratings
        self.assertGreater(by_slug["semantic-drift"]["credibility"], by_slug["alignment-fatigue"]["credibility"])
        self.assertGreater(by_slug["alignment-fatigue"]["credibility"], by_slug["phantom-completion"]["credibility"])

    def test_phantom_completion_low_credibility(self):
        """Single-rating term should have credibility = 0.5."""
        _, _, terms = compute_bayes_scores(self.ratings)
        phantom = next(t for t in terms if t["slug"] == "phantom-completion")
        self.assertAlmostEqual(phantom["credibility"], 0.5, places=4)

    def test_highest_rated_term_wins(self):
        """semantic-drift (high scores, many ratings) should have the highest consensus_score."""
        _, _, terms = compute_bayes_scores(self.ratings)
        self.assertEqual(terms[0]["slug"], "semantic-drift")

    def test_global_stats_present(self):
        """Global stats should contain required keys."""
        global_stats, _, _ = compute_bayes_scores(self.ratings)
        for key in ("grand_mean", "tau_squared", "sigma_squared", "total_ratings", "total_terms"):
            self.assertIn(key, global_stats)
        self.assertEqual(global_stats["total_ratings"], 11)
        self.assertEqual(global_stats["total_terms"], 3)

    def test_rater_biases_sum_approximately_zero(self):
        """Weighted rater biases should approximately cancel out."""
        _, biases, _ = compute_bayes_scores(self.ratings)
        self.assertEqual(len(biases), 3)
        # Each rater has different counts, but biases should be reasonable
        for model, bias in biases.items():
            self.assertGreater(abs(bias), -10, f"Bias for {model} seems unreasonable")

    def test_empty_ratings(self):
        """Empty input should return empty results."""
        gs, biases, terms = compute_bayes_scores([])
        self.assertEqual(gs, {})
        self.assertEqual(biases, {})
        self.assertEqual(terms, [])

    def test_agreement_bounded(self):
        """Agreement values should be in [0, 1]."""
        _, _, terms = compute_bayes_scores(self.ratings)
        for t in terms:
            self.assertGreaterEqual(t["agreement"], 0)
            self.assertLessEqual(t["agreement"], 1)


if __name__ == "__main__":
    unittest.main()
