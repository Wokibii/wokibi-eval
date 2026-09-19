"""
Test module for ranking evaluation functions.
"""

from wokibi_eval.evaluation.ranking import mean_average_precision_at_k, recall_at_k, _preparar_dados_para_metrica
import pytest
import numpy as np


class TestMeanAveragePrecisionAtK:
    """Test cases for mean_average_precision_at_k function."""

    def test_basic_functionality(self):
        """Test basic MAP@k calculation."""
        y_true = [[1, 2, 3], [2, 4], [1, 3, 5]]
        y_pred = [[1, 2, 4, 5], [2, 3, 4, 6], [1, 2, 3, 4]]

        map_2 = mean_average_precision_at_k(y_true, y_pred, k=2)
        assert isinstance(map_2, float)
        assert 0.0 <= map_2 <= 1.0

    def test_k_zero_uses_all_predictions(self):
        """Test that k=0 uses all predictions."""
        y_true = [[1, 2, 3], [2, 4]]
        y_pred = [[1, 2, 4, 5], [2, 3, 4, 6]]

        map_all = mean_average_precision_at_k(y_true, y_pred, k=0)
        map_4 = mean_average_precision_at_k(y_true, y_pred, k=4)

        # Should be the same when k=0 and k=len(predictions)
        assert abs(map_all - map_4) < 0.001

    def test_empty_y_true(self):
        """Test with empty y_true."""
        result = mean_average_precision_at_k([], [[1, 2, 3]], k=2)
        assert result == 0.0

    def test_negative_k(self):
        """Test with negative k (should behave like k=0)."""
        y_true = [[1, 2, 3]]
        y_pred = [[1, 2, 4]]
        result_negative = mean_average_precision_at_k(y_true, y_pred, k=-1)
        result_zero = mean_average_precision_at_k(y_true, y_pred, k=0)
        # Negative k should behave the same as k=0 (use all predictions)
        assert abs(result_negative - result_zero) < 0.001

    def test_1d_arrays(self):
        """Test with 1D arrays."""
        y_true = [1, 2, 3]
        y_pred = [1, 2, 4, 5]
        result = mean_average_precision_at_k(y_true, y_pred, k=2)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_numpy_arrays(self):
        """Test with NumPy arrays."""
        y_true = np.array([[1, 2], [3, 4]])
        y_pred = np.array([[1, 3, 5], [3, 4, 6]])
        result = mean_average_precision_at_k(y_true, y_pred, k=2)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_no_relevant_items(self):
        """Test when user has no relevant items."""
        y_true = [[1, 2], [], [3, 4]]
        y_pred = [[1, 2, 3], [1, 2, 3], [3, 4, 5]]
        result = mean_average_precision_at_k(y_true, y_pred, k=2)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_perfect_ranking(self):
        """Test with perfect ranking."""
        y_true = [[1, 2, 3]]
        y_pred = [[1, 2, 3, 4]]
        result = mean_average_precision_at_k(y_true, y_pred, k=3)
        assert result == 1.0

    def test_no_relevant_items_found(self):
        """Test when no relevant items are found."""
        y_true = [[1, 2, 3]]
        y_pred = [[4, 5, 6]]
        result = mean_average_precision_at_k(y_true, y_pred, k=3)
        assert result == 0.0


class TestRecallAtK:
    """Test cases for recall_at_k function."""

    def test_basic_functionality(self):
        """Test basic Recall@k calculation."""
        y_true = [[1, 2, 3], [2, 4], [1, 3, 5]]
        y_pred = [[1, 2, 4, 5], [2, 3, 4, 6], [1, 2, 3, 4]]

        recall_2 = recall_at_k(y_true, y_pred, k=2)
        assert isinstance(recall_2, float)
        assert 0.0 <= recall_2 <= 1.0

    def test_k_zero_uses_all_predictions(self):
        """Test that k=0 uses all predictions."""
        y_true = [[1, 2, 3], [2, 4]]
        y_pred = [[1, 2, 4, 5], [2, 3, 4, 6]]

        recall_all = recall_at_k(y_true, y_pred, k=0)
        recall_4 = recall_at_k(y_true, y_pred, k=4)

        # Should be the same when k=0 and k=len(predictions)
        assert abs(recall_all - recall_4) < 0.001

    def test_empty_y_true(self):
        """Test with empty y_true."""
        result = recall_at_k([], [[1, 2, 3]], k=2)
        assert result == 0.0

    def test_negative_k(self):
        """Test with negative k (should behave like k=0)."""
        y_true = [[1, 2, 3]]
        y_pred = [[1, 2, 4]]
        result_negative = recall_at_k(y_true, y_pred, k=-1)
        result_zero = recall_at_k(y_true, y_pred, k=0)
        # Negative k should behave the same as k=0 (use all predictions)
        assert abs(result_negative - result_zero) < 0.001

    def test_1d_arrays(self):
        """Test with 1D arrays."""
        y_true = [1, 2, 3]
        y_pred = [1, 2, 4, 5]
        result = recall_at_k(y_true, y_pred, k=2)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_numpy_arrays(self):
        """Test with NumPy arrays."""
        y_true = np.array([[1, 2], [3, 4]])
        y_pred = np.array([[1, 3, 5], [3, 4, 6]])
        result = recall_at_k(y_true, y_pred, k=2)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_no_relevant_items(self):
        """Test when user has no relevant items."""
        y_true = [[1, 2], [], [3, 4]]
        y_pred = [[1, 2, 3], [1, 2, 3], [3, 4, 5]]
        result = recall_at_k(y_true, y_pred, k=2)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_perfect_recall(self):
        """Test with perfect recall."""
        y_true = [[1, 2, 3]]
        y_pred = [[1, 2, 3, 4]]
        result = recall_at_k(y_true, y_pred, k=3)
        assert result == 1.0

    def test_no_relevant_items_found(self):
        """Test when no relevant items are found."""
        y_true = [[1, 2, 3]]
        y_pred = [[4, 5, 6]]
        result = recall_at_k(y_true, y_pred, k=3)
        assert result == 0.0

    def test_recall_increases_with_k(self):
        """Test that recall generally increases with k."""
        y_true = [[1, 2, 3], [2, 4]]
        y_pred = [[1, 2, 4, 5], [2, 3, 4, 6]]

        recall_1 = recall_at_k(y_true, y_pred, k=1)
        recall_2 = recall_at_k(y_true, y_pred, k=2)
        recall_3 = recall_at_k(y_true, y_pred, k=3)

        assert recall_2 >= recall_1, "Recall@2 should be >= Recall@1"
        assert recall_3 >= recall_2, "Recall@3 should be >= Recall@2"

    def test_specific_calculation(self):
        """Test specific recall calculation."""
        y_true = [[1, 2, 3], [2, 4], [1, 3, 5]]
        y_pred = [[1, 2, 4, 5], [2, 3, 4, 6], [1, 2, 3, 4]]

        # Manual calculation for k=2:
        # User 0: Found [1, 2] out of [1, 2, 3] = 2/3 = 0.667
        # User 1: Found [2] out of [2, 4] = 1/2 = 0.5
        # User 2: Found [1] out of [1, 3, 5] = 1/3 = 0.333
        # Average: (0.667 + 0.5 + 0.333) / 3 = 0.5
        expected_recall = (2/3 + 1/2 + 1/3) / 3
        actual_recall = recall_at_k(y_true, y_pred, k=2)

        assert abs(actual_recall - expected_recall) < 0.001


class TestHelperFunction:
    """Test cases for the helper function _preparar_dados_para_metrica."""

    def test_empty_input(self):
        """Test with empty input."""
        result = _preparar_dados_para_metrica([])
        assert result == []

    def test_single_list(self):
        """Test with single list."""
        result = _preparar_dados_para_metrica([1, 2, 3])
        assert result == [{1, 2, 3}]

    def test_nested_lists(self):
        """Test with nested lists."""
        result = _preparar_dados_para_metrica([[1, 2], [3, 4]])
        assert result == [{1, 2}, {3, 4}]

    def test_numpy_array(self):
        """Test with numpy array."""
        arr = np.array([[1, 2], [3, 4]])
        result = _preparar_dados_para_metrica(arr)
        assert result == [{1, 2}, {3, 4}]

    def test_mixed_types(self):
        """Test with mixed data types."""
        result = _preparar_dados_para_metrica([["a", "b"], [1, 2]])
        assert result == [{"a", "b"}, {1, 2}]


class TestDefaultBehavior:
    """Test cases for default k=0 behavior."""

    def test_default_k_zero_map(self):
        """Test that default k=0 works for MAP."""
        y_true = [[1, 2, 3], [2, 4]]
        y_pred = [[1, 2, 4, 5], [2, 3, 4, 6]]

        # Test default behavior (k=0)
        result_default = mean_average_precision_at_k(y_true, y_pred)
        # Test explicit k=0
        result_explicit = mean_average_precision_at_k(y_true, y_pred, k=0)

        assert abs(result_default - result_explicit) < 0.001
        assert isinstance(result_default, float)
        assert 0.0 <= result_default <= 1.0

    def test_default_k_zero_recall(self):
        """Test that default k=0 works for Recall."""
        y_true = [[1, 2, 3], [2, 4]]
        y_pred = [[1, 2, 4, 5], [2, 3, 4, 6]]

        # Test default behavior (k=0)
        result_default = recall_at_k(y_true, y_pred)
        # Test explicit k=0
        result_explicit = recall_at_k(y_true, y_pred, k=0)

        assert abs(result_default - result_explicit) < 0.001
        assert isinstance(result_default, float)
        assert 0.0 <= result_default <= 1.0


class TestRankingIntegration:
    """Integration tests for ranking functions."""

    def test_both_functions_work_together(self):
        """Test that both functions work with the same data."""
        y_true = [[1, 2, 3], [2, 4], [1, 3, 5]]
        y_pred = [[1, 2, 4, 5], [2, 3, 4, 6], [1, 2, 3, 4]]

        map_result = mean_average_precision_at_k(y_true, y_pred, k=2)
        recall_result = recall_at_k(y_true, y_pred, k=2)

        assert isinstance(map_result, float)
        assert isinstance(recall_result, float)
        assert 0.0 <= map_result <= 1.0
        assert 0.0 <= recall_result <= 1.0

    def test_consistency_with_different_k_values(self):
        """Test consistency with different k values."""
        y_true = [[1, 2, 3, 4], [2, 4, 5]]
        y_pred = [[1, 2, 3, 4, 5], [2, 4, 5, 6, 7]]

        # Test multiple k values
        for k in [1, 2, 3, 4, 5]:
            map_k = mean_average_precision_at_k(y_true, y_pred, k=k)
            recall_k = recall_at_k(y_true, y_pred, k=k)

            assert isinstance(map_k, float)
            assert isinstance(recall_k, float)
            assert 0.0 <= map_k <= 1.0
            assert 0.0 <= recall_k <= 1.0

    def test_edge_case_empty_predictions(self):
        """Test with empty predictions."""
        y_true = [[1, 2, 3]]
        y_pred = [[]]

        map_result = mean_average_precision_at_k(y_true, y_pred, k=2)
        recall_result = recall_at_k(y_true, y_pred, k=2)

        assert map_result == 0.0
        assert recall_result == 0.0

    def test_hashable_items(self):
        """Test with different types of hashable items."""
        # Test with strings
        y_true_str = [["a", "b", "c"], ["b", "d"]]
        y_pred_str = [["a", "b", "d", "e"], ["b", "c", "d", "f"]]

        map_str = mean_average_precision_at_k(y_true_str, y_pred_str, k=2)
        recall_str = recall_at_k(y_true_str, y_pred_str, k=2)

        assert isinstance(map_str, float)
        assert isinstance(recall_str, float)

        # Test with mixed types
        y_true_mixed = [[1, "a", 2.5], ["b", 3]]
        y_pred_mixed = [[1, "a", 2.5, "c"], ["b", 3, 4, "d"]]

        map_mixed = mean_average_precision_at_k(
            y_true_mixed, y_pred_mixed, k=2)
        recall_mixed = recall_at_k(y_true_mixed, y_pred_mixed, k=2)

        assert isinstance(map_mixed, float)
        assert isinstance(recall_mixed, float)
