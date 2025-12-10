import pytest
import pandas as pd
import numpy as np
from eigenfactor import calculate_eigenfactor


def test_check_ring(result):
    scores = result.set_index("journal_id")["eigenfactor_score"]
    assert np.isclose(scores[1], 33.333, atol=0.1)
    assert np.isclose(scores[2], 33.333, atol=0.1)
    assert np.isclose(scores[3], 33.333, atol=0.1)


def test_check_self_citation(result):
    scores = result.set_index("journal_id")["eigenfactor_score"]
    # 2 should have higher score than 1 because it receives explicit citations
    # 1 receives influence only from teleportation/dangling redistribution
    assert scores[2] > scores[1]


def test_check_disconnected(result):
    scores = result.set_index("journal_id")["eigenfactor_score"]
    assert np.allclose(scores.values, 25.0)


@pytest.mark.parametrize(
    "name, citations_data, journal_article_counts, expected_len, expected_sum, custom_check",
    [
        (
            "simple_ring",
            {
                "citing_journal": [1, 2, 3],
                "cited_journal": [2, 3, 1],
                "citation_count": [10, 10, 10],
            },
            {1: 10, 2: 10, 3: 10},
            3,
            100.0,
            test_check_ring,
        ),
        (
            "dangling_node",
            {"citing_journal": [1], "cited_journal": [2], "citation_count": [10]},
            {1: 100, 2: 100},
            2,
            100.0,
            None,
        ),
        (
            "self_citation",
            {
                "citing_journal": [1, 1],
                "cited_journal": [1, 2],
                "citation_count": [100, 10],
            },
            {1: 10, 2: 10},
            2,
            100.0,
            test_check_self_citation,
        ),
        (
            "disconnected",
            {
                "citing_journal": [1, 2, 3, 4],
                "cited_journal": [2, 1, 4, 3],
                "citation_count": [10, 10, 10, 10],
            },
            {1: 10, 2: 10, 3: 10, 4: 10},
            4,
            100.0,
            test_check_disconnected,
        ),
        (
            "empty",
            {"citing_journal": [], "cited_journal": [], "citation_count": []},
            {},
            0,
            0.0,
            None,
        ),
    ],
)
def test_calculate_eigenfactor(
    name,
    citations_data,
    journal_article_counts,
    expected_len,
    expected_sum,
    custom_check,
):
    """
    Parametrized test for calculate_eigenfactor covering various scenarios.
    """
    # Handle empty dataframe creation with correct columns
    if not citations_data["citing_journal"]:
        citations_df = pd.DataFrame(
            columns=["citing_journal", "cited_journal", "citation_count"]
        )
    else:
        citations_df = pd.DataFrame(citations_data)

    result = calculate_eigenfactor(citations_df, journal_article_counts)

    assert len(result) == expected_len
    assert "journal_id" in result.columns
    assert "eigenfactor_score" in result.columns

    if not result.empty:
        assert np.isclose(result["eigenfactor_score"].sum(), expected_sum)
    else:
        assert expected_sum == 0.0

    if custom_check:
        custom_check(result)
