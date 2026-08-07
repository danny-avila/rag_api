"""Unit tests for the fast-v1 embed-blend scoring primitives."""

import math

from app.services.rerank import (
    BlendConfig,
    blend_scores,
    cosine_similarity,
    order_by_score,
    rank_positions,
)

CONFIG = BlendConfig(k=60, similarity_weight=1.0, base_weight=1.0)


class TestCosineSimilarity:
    def test_identical_direction_is_one(self):
        assert math.isclose(cosine_similarity([1.0, 0.0], [3.0, 0.0]), 1.0)

    def test_orthogonal_is_zero(self):
        assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 5.0]), 0.0)

    def test_opposite_direction_is_negative_one(self):
        assert math.isclose(cosine_similarity([1.0, 0.0], [-2.0, 0.0]), -1.0)

    def test_zero_vector_has_no_similarity(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) is None

    def test_mismatched_width_has_no_similarity(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) is None


class TestRankPositions:
    def test_highest_value_ranks_first(self):
        assert rank_positions([0.1, 0.9, 0.5]) == [3, 1, 2]

    def test_ties_break_on_request_position(self):
        assert rank_positions([0.5, 0.5, 0.5]) == [1, 2, 3]

    def test_unscored_entries_rank_last_in_order(self):
        assert rank_positions([None, 0.2, None, 0.9]) == [3, 2, 4, 1]


class TestBlendScores:
    def test_blend_is_not_pure_embedding_order(self):
        """The whole point of the blend: base_score can overturn cosine order."""
        similarities = [1.0, 0.9, 0.8]
        base_scores = [1.0, 5.0, 10.0]
        scores = blend_scores(similarities, base_scores, CONFIG)
        pure = order_by_score(blend_scores(similarities, [None, None, None], CONFIG), 3)
        blended = order_by_score(scores, 3)
        assert [item.index for item in pure] == [0, 1, 2]
        assert [item.index for item in blended] == [0, 2, 1]

    def test_a_missing_arm_is_dropped_rather_than_treated_as_a_constant(self):
        base_only = blend_scores([None, None, None], [1.0, 3.0, 2.0], CONFIG)
        assert [item.index for item in order_by_score(base_only, 3)] == [1, 2, 0]

    def test_candidates_without_vectors_still_rank_through_the_base_arm(self):
        """A vectorless candidate is not buried: a strong base score lifts it."""
        scores = blend_scores([0.9, None, 0.8], [3.0, 100.0, 1.0], CONFIG)
        assert scores[1] > 0.0
        assert [item.index for item in order_by_score(scores, 3)] == [0, 1, 2]

    def test_exact_ties_break_on_request_position(self):
        similarities = [1.0, 0.9, 0.8]
        base_scores = [1.0, 5.0, 10.0]
        scores = blend_scores(similarities, base_scores, CONFIG)
        assert scores[0] == scores[2]
        assert [item.index for item in order_by_score(scores, 3)] == [0, 2, 1]

    def test_identical_inputs_produce_identical_output(self):
        similarities = [0.4, 0.4, 0.4, 0.4]
        base_scores = [1.0, 1.0, 1.0, 1.0]
        first = order_by_score(blend_scores(similarities, base_scores, CONFIG), 4)
        second = order_by_score(blend_scores(similarities, base_scores, CONFIG), 4)
        assert [item.index for item in first] == [item.index for item in second]
        assert [item.index for item in first] == [0, 1, 2, 3]


class TestOrderByScore:
    def test_truncates_to_top_n(self):
        ranked = order_by_score([0.1, 0.5, 0.3, 0.9], 2)
        assert [item.index for item in ranked] == [3, 1]

    def test_weights_are_configurable(self):
        similarity_heavy = BlendConfig(k=60, similarity_weight=10.0, base_weight=1.0)
        scores = blend_scores([1.0, 0.9, 0.8], [1.0, 5.0, 10.0], similarity_heavy)
        assert [item.index for item in order_by_score(scores, 3)] == [0, 1, 2]
