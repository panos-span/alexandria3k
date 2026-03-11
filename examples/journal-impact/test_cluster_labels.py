#
# Alexandria3k Crossref bibliographic metadata processing
# Copyright (C) 2026  Panagiotis Spanakis
# SPDX-License-Identifier: GPL-3.0-or-later
#

"""Tests for community label generation."""

import pandas as pd

from cluster_labels import generate_cluster_labels, normalize_publisher_name


def test_normalize_publisher_name_strips_corporate_suffixes():
    assert normalize_publisher_name("Agronomy Society Press") == "Agronomy Society"
    assert normalize_publisher_name("Nature Publishing Group") == "Nature"
    assert normalize_publisher_name("") == ""


def test_generate_cluster_labels_prefers_title_terms():
    cluster_df = pd.DataFrame(
        [
            {
                "journal_id": 1,
                "community_id": 1,
                "weight": 1.0,
                "title": "Applied Physics Materials",
                "publisher": "Alpha Press",
                "context_impact": 2.0,
                "prestige_rank": 1.0,
                "network_centrality": 5.0,
            },
            {
                "journal_id": 2,
                "community_id": 1,
                "weight": 1.0,
                "title": "Materials Physics Letters",
                "publisher": "Beta Press",
                "context_impact": 1.5,
                "prestige_rank": 1.0,
                "network_centrality": 4.0,
            },
        ]
    )

    labels_df = generate_cluster_labels(cluster_df)
    label = labels_df.loc[0, "label"].lower()
    assert "materials" in label or "physics" in label
    assert labels_df.loc[0, "top_phrases"]


def test_generate_cluster_labels_falls_back_to_publisher():
    cluster_df = pd.DataFrame(
        [
            {
                "journal_id": 1,
                "community_id": 2,
                "weight": 1.0,
                "title": "T1",
                "publisher": "Agronomy Society Press",
                "context_impact": 0.5,
                "prestige_rank": 0.0,
                "network_centrality": 0.0,
            }
        ]
    )

    labels_df = generate_cluster_labels(cluster_df)
    assert "Agronomy" in labels_df.loc[0, "label"]


def test_generate_cluster_labels_produces_different_labels_for_different_communities():
    cluster_df = pd.DataFrame(
        [
            {
                "journal_id": 1,
                "community_id": 1,
                "weight": 1.0,
                "title": "Applied Physics Materials",
                "publisher": "Alpha Press",
                "context_impact": 2.0,
                "prestige_rank": 1.0,
                "network_centrality": 5.0,
            },
            {
                "journal_id": 2,
                "community_id": 2,
                "weight": 1.0,
                "title": "Applied Chemistry Synthesis",
                "publisher": "Beta Press",
                "context_impact": 2.0,
                "prestige_rank": 1.0,
                "network_centrality": 5.0,
            },
        ]
    )

    labels_df = generate_cluster_labels(cluster_df)
    assert labels_df.loc[0, "label"] != labels_df.loc[1, "label"]
    assert "physics" in labels_df.loc[0, "label"].lower()
    assert "chemistry" in labels_df.loc[1, "label"].lower()


def test_labels_are_title_cased():
    cluster_df = pd.DataFrame(
        [
            {
                "journal_id": 1,
                "community_id": 1,
                "weight": 1.0,
                "title": "Organic Chemistry Methods",
                "publisher": "Alpha Press",
                "context_impact": 1.0,
                "prestige_rank": 0.5,
                "network_centrality": 2.0,
            },
        ]
    )

    labels_df = generate_cluster_labels(cluster_df)
    label = labels_df.loc[0, "label"]
    # Each word in the label should be title-cased
    for word in label.replace("&", "").split():
        assert word[0].isupper(), f"'{word}' is not title-cased"