#!/mnt/c/Users/panagiotis/Desktop/GitHub/alexandria3k/examples/journal-impact/.venv/bin/python3
#
# Alexandria3k Crossref bibliographic metadata processing
# Copyright (C) 2026  Panagiotis Spanakis
# SPDX-License-Identifier: GPL-3.0-or-later
#

"""Generate human-readable labels for journal communities.

Labels are derived using TF-IDF analysis across community-level
pseudo-documents constructed from journal titles.  For each community
a weighted corpus is built by repeating journal titles in proportion
to each journal's importance score.  Scikit-learn's TfidfVectorizer
then identifies the terms most distinctive to each community relative
to all others.  The top-scoring non-redundant terms become the
community label.

When title-derived terms are insufficient the label falls back to
a publisher-based summary.
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import sqlite3
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Maximum number of label terms per community
MAX_LABEL_TERMS = 2

# Legal and corporate suffixes removed from publisher names
PUBLISHER_SUFFIXES = frozenset({
    "inc", "ltd", "llc", "co", "corp", "gmbh", "sa", "ag",
    "press", "publishing", "publishers", "publisher",
    "group", "holdings",
})


def normalize_text(value: str) -> str:
    """Return a normalized lowercase representation of *value*."""
    if not value:
        return ""
    return unicodedata.normalize("NFKC", value).casefold()


def journal_importance(row: pd.Series) -> float:
    """Compute a bounded importance score for corpus weighting."""
    score = 1.0
    score += min(float(row.get("context_impact", 0.0) or 0.0), 5.0)
    score += min(float(row.get("prestige_rank", 0.0) or 0.0), 5.0)
    score += min(float(row.get("network_centrality", 0.0) or 0.0) / 10.0, 5.0)
    return score * float(row.get("weight", 1.0) or 1.0)


def build_community_corpus(working_df: pd.DataFrame) -> dict[int, str]:
    """Build one pseudo-document per community from weighted journal titles.

    Each journal's title is repeated in proportion to its importance
    score so that influential journals contribute more to the term
    frequency computation.
    """
    corpus: dict[int, list[str]] = collections.defaultdict(list)
    for row in working_df.to_dict("records"):
        community_id = int(row["community_id"])
        title = normalize_text(row.get("title", ""))
        if not title:
            continue
        repeats = max(1, round(float(row.get("importance", 1.0))))
        corpus[community_id].extend([title] * repeats)
    return {cid: " ".join(titles) for cid, titles in corpus.items()}


def select_nonredundant_terms(
    term_scores: list[tuple[str, float]],
    limit: int = MAX_LABEL_TERMS,
) -> list[str]:
    """Select high-scoring terms avoiding token-level redundancy."""
    selected: list[str] = []
    selected_tokens: set[str] = set()

    for term, _ in term_scores:
        tokens = set(term.split())
        if tokens <= selected_tokens:
            continue
        if len(tokens) == 1 and tokens & selected_tokens:
            continue
        selected.append(term)
        selected_tokens.update(tokens)
        if len(selected) >= limit:
            break

    return selected


def format_label(terms: list[str]) -> str:
    """Format selected terms into a publication-quality title-cased label."""
    return " & ".join(term.title() for term in terms)


def normalize_publisher_name(publisher: str) -> str:
    """Return a concise publisher name stripped of corporate suffixes."""
    if not publisher:
        return ""
    tokens = normalize_text(publisher).split()
    filtered = [
        t for t in tokens
        if t not in PUBLISHER_SUFFIXES and len(t) >= 3
    ]
    if filtered:
        return " ".join(filtered[:4]).title()
    return publisher.strip().title()


def generate_cluster_labels(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """Generate labels and supporting metadata for each community.

    Uses TF-IDF analysis to identify terms that are distinctive to each
    community relative to all others.  Journal importance weights influence
    the term frequency computation through title repetition in the corpus.
    """
    working_df = cluster_df.copy()
    working_df["importance"] = working_df.apply(journal_importance, axis=1)

    community_corpus = build_community_corpus(working_df)
    community_ids = sorted(community_corpus.keys())
    documents = [community_corpus[cid] for cid in community_ids]

    # Fit TF-IDF across all community documents
    tfidf_matrix = None
    feature_names = np.array([])

    if documents:
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            stop_words="english",
            token_pattern=r"(?u)\b[^\W\d_]{3,}\b",
            sublinear_tf=True,
            min_df=1,
            max_df=0.85 if len(documents) >= 2 else 1.0,
            use_idf=len(documents) >= 2,
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
        except ValueError:
            pass  # Empty vocabulary; fall through to publisher labels

    labels = []

    for idx, community_id in enumerate(community_ids):
        community_rows = working_df[
            working_df["community_id"] == community_id
        ]

        # Extract distinctive terms from TF-IDF
        top_terms: list[str] = []
        if tfidf_matrix is not None and len(feature_names) > 0:
            scores = tfidf_matrix.getrow(idx).toarray().flatten()
            ranked_indices = np.argsort(scores)[::-1]
            term_scores: list[tuple[str, float]] = [
                (str(feature_names[i]), float(scores[i]))
                for i in ranked_indices
                if scores[i] > 0
            ]
            top_terms = select_nonredundant_terms(term_scores)

        label = format_label(top_terms)

        # Fallback: publisher-based label
        if not label:
            pub_scores: collections.defaultdict[str, float] = (
                collections.defaultdict(float)
            )
            for row in community_rows.to_dict("records"):
                pub = normalize_publisher_name(row.get("publisher", ""))
                if pub:
                    pub_scores[pub] += float(
                        row.get("importance", 1.0) or 1.0
                    )
            top_publishers = sorted(
                pub_scores, key=lambda p: -pub_scores[p]
            )[:2]
            label = " & ".join(top_publishers) if top_publishers else ""

        if not label:
            label = f"Community {community_id}"

        # Collect metadata
        pub_meta: collections.defaultdict[str, float] = (
            collections.defaultdict(float)
        )
        ranked_rows: list[tuple[float, dict]] = []
        for row in community_rows.to_dict("records"):
            pub = normalize_publisher_name(row.get("publisher", ""))
            if pub:
                pub_meta[pub] += float(row.get("importance", 1.0) or 1.0)
            ranked_rows.append(
                (float(row.get("importance", 1.0) or 1.0), row)
            )

        ranked_rows.sort(
            key=lambda item: (-item[0], item[1].get("title", ""))
        )
        sample_journals = [
            row["title"]
            for _, row in ranked_rows[:3]
            if row.get("title")
        ]
        ranked_publishers = sorted(
            pub_meta, key=lambda p: -pub_meta[p]
        )[:3]

        labels.append(
            {
                "community_id": int(community_id),
                "label": label,
                "top_phrases": ", ".join(t.title() for t in top_terms),
                "top_publishers": ", ".join(ranked_publishers),
                "sample_journals": " | ".join(sample_journals),
                "journals_count": int(
                    community_rows["journal_id"].nunique()
                ),
            }
        )

    return (
        pd.DataFrame(labels)
        .sort_values("community_id")
        .reset_index(drop=True)
    )


def get_db_connection(db_path: str, rolap_db_path: str) -> sqlite3.Connection:
    """Connect to the main database and attach the ROLAP database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file '{db_path}' not found.")
    if not os.path.exists(rolap_db_path):
        raise FileNotFoundError(f"Database file '{rolap_db_path}' not found.")

    conn = sqlite3.connect(db_path)
    conn.execute(f"ATTACH DATABASE '{rolap_db_path}' AS rolap")
    return conn


def load_cluster_source_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load the data used to derive community labels."""
    query = """
        SELECT
            jc.journal_id,
            jc.community_id,
            jc.weight,
            jn.title,
            jn.publisher,
            COALESCE(ci.impact_score, 0.0) AS context_impact,
            COALESCE(pr.prestige_score, 0.0) AS prestige_rank,
            COALESCE(nc.centrality_score, 0.0) AS network_centrality
        FROM rolap.journal_communities jc
        LEFT JOIN journal_names jn
            ON jn.id = jc.journal_id
        LEFT JOIN rolap.context_impact ci
            ON ci.journal_id = jc.journal_id
        LEFT JOIN rolap.prestige_rank pr
            ON pr.journal_id = jc.journal_id
        LEFT JOIN rolap.network_centrality nc
            ON nc.journal_id = jc.journal_id
    """
    return pd.read_sql_query(query, conn)


def write_df_to_attached(conn: sqlite3.Connection,
                         df: pd.DataFrame,
                         table: str,
                         db: str = "rolap") -> None:
    """Drop and recreate table ``db.table`` and load DataFrame contents."""
    type_map = {
        "int64": "INTEGER",
        "float64": "REAL",
        "bool": "INTEGER",
        "object": "TEXT",
    }
    columns = list(df.columns)
    definitions = [
        f'"{column}" {type_map.get(str(df[column].dtype), "TEXT")}'
        for column in columns
    ]

    conn.execute(f'DROP TABLE IF EXISTS {db}."{table}"')
    conn.execute(f'CREATE TABLE {db}."{table}" ({", ".join(definitions)})')

    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    insert_sql = (
        f'INSERT INTO {db}."{table}" ({quoted_columns}) VALUES ({placeholders})'
    )
    conn.executemany(insert_sql, df.itertuples(index=False, name=None))
    conn.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate human-readable labels for journal communities."
    )
    parser.add_argument("--db", required=True, help="Path to the main SQLite database")
    parser.add_argument("--rolap-db", required=True, help="Path to the ROLAP SQLite database")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conn = None
    try:
        conn = get_db_connection(args.db, args.rolap_db)
        cluster_df = load_cluster_source_data(conn)
        labels_df = generate_cluster_labels(cluster_df)
        write_df_to_attached(conn, labels_df, "cluster_labels")
        logging.info("Saved %d cluster labels to rolap.cluster_labels", len(labels_df))
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()