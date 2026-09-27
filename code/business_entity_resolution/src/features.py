"""Vectorized feature extraction for entity pairs using RapidFuzz C++ kernels."""
from __future__ import annotations

from rapidfuzz import fuzz

FEATURE_NAMES = [
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_ratio",
    "name_partial_ratio",
    "name_sim_max",
    "has_empty_addr",
    "addr_token_set_ratio",
    "addr_token_sort_ratio",
    "addr_ratio",
    "num_common",
    "num_jaccard",
    "exact_name",
    "exact_addr",
    "block_weight",
    "name_addr_interaction",
]


def extract_pair_features(
    qn: str,
    qa: str,
    q_nums: set[str],
    tn: str,
    ta: str,
    t_nums: set[str],
    block_weight: int,
) -> list[float]:
    """Computes a rich 15-dimensional feature vector for a candidate pair."""
    # Name features
    n_set = fuzz.token_set_ratio(qn, tn) / 100.0
    n_sort = fuzz.token_sort_ratio(qn, tn) / 100.0
    n_ratio = fuzz.ratio(qn, tn) / 100.0
    n_partial = fuzz.partial_ratio(qn, tn) / 100.0
    name_sim_max = max(n_set, n_sort, n_ratio, n_partial)

    # Address features
    has_empty_addr = 1.0 if (not qa or not ta) else 0.0
    if not has_empty_addr:
        a_set = fuzz.token_set_ratio(qa, ta) / 100.0
        a_sort = fuzz.token_sort_ratio(qa, ta) / 100.0
        a_ratio = fuzz.ratio(qa, ta) / 100.0
    else:
        a_set = 0.0
        a_sort = 0.0
        a_ratio = 0.0

    # Cross-field interaction.
    # IMPORTANT: when address is missing we must NOT assume high similarity —
    # doing so undoes the whole point of this feature (suppressing false merges
    # on weak/incomplete records). Use a neutral midpoint instead.
    effective_addr = a_set if not has_empty_addr else 0.5
    name_addr_interaction = name_sim_max * effective_addr

    # Number overlap features
    common_nums = len(q_nums & t_nums)
    total_nums = len(q_nums | t_nums)
    num_jaccard = (common_nums / total_nums) if total_nums > 0 else 0.0

    exact_name = 1.0 if qn and qn == tn else 0.0
    exact_addr = 1.0 if qa and qa == ta else 0.0

    return [
        n_set,
        n_sort,
        n_ratio,
        n_partial,
        name_sim_max,
        has_empty_addr,
        a_set,
        a_sort,
        a_ratio,
        float(common_nums),
        float(num_jaccard),
        exact_name,
        exact_addr,
        float(block_weight),
        name_addr_interaction,
    ]