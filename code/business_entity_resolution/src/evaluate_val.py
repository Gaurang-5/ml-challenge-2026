"""Evaluates Macro F0.5 on a truly held-out training tail, using the trained LightGBM model.

Uses the LAST --val-holdout rows of train_ground_truth.tsv, mirroring train_model.py's
reservation of that same tail — so this script and train_model.py can never overlap as
long as both are run with the same --val-holdout value.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import sys
import time
import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from normalizer import clean_business_name, clean_address
from blocking import generate_blocking_keys, InvertedIndex
from features import extract_pair_features

csv.field_size_limit(sys.maxsize)

# Canonical blocking parameters — MUST match train_model.py and entity_resolution.py.
TOP_K = 60
MAX_KEY_FREQUENCY = 1500
MATCH_CAP = 12  # must match entity_resolution.py


def evaluate(
    train_dir: Path,
    model_path: Path,
    val_holdout: int = 15000,
    threshold: float = 0.60,
    distractor_cap: int = 1_000_000,
) -> None:
    print(f"Loading LightGBM model from {model_path}...", flush=True)
    booster = lgb.Booster(model_file=str(model_path))

    print(f"Loading ground truth tail ({val_holdout:,} rows, held out from training)...", flush=True)
    with open(train_dir / "train_ground_truth.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        all_rows = list(reader)

    total_rows = len(all_rows)
    if val_holdout >= total_rows:
        raise ValueError(f"val_holdout ({val_holdout}) must be smaller than total rows ({total_rows})")

    val_rows = all_rows[total_rows - val_holdout:]
    val_gt: dict[str, set[str]] = {
        row["source1_entity_id"]: set(m for m in row["matched_entity_ids"].split(",") if m)
        for row in val_rows
    }

    needed_s1 = set(val_gt.keys())
    needed_targets = {m for ms in val_gt.values() for m in ms}
    print(f"Held-out queries: {len(needed_s1):,}, True target matches: {len(needed_targets):,}.", flush=True)

    queries: list[tuple[str, str, str, set[str]]] = []
    with open(train_dir / "train_source1.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row["entity_id"]
            if eid in needed_s1:
                qn, _ = clean_business_name(row["business_name"])
                qa, _, q_nums = clean_address(row["business_address"])
                queries.append((eid, qn, qa, set(q_nums)))
            if len(queries) == len(needed_s1):
                break

    print(f"Building target index with distractor_cap={distractor_cap:,}...", flush=True)
    target_ids: list[str] = []
    target_records: list[tuple[str, str, set[str]]] = []
    index = InvertedIndex()

    distractors_added = 0
    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        with open(train_dir / fname, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                eid = row["entity_id"]
                is_needed = eid in needed_targets
                if is_needed or (distractors_added < distractor_cap):
                    if not is_needed:
                        distractors_added += 1
                    t_idx = len(target_ids)
                    target_ids.append(eid)
                    tn, _ = clean_business_name(row["business_name"])
                    ta, _, t_nums = clean_address(row["business_address"])
                    target_records.append((tn, ta, set(t_nums)))
                    keys = generate_blocking_keys(row["business_name"], row["business_address"])
                    index.add(t_idx, keys)

    print(f"Total targets indexed: {len(target_ids):,}. Running inference...", flush=True)
    t0 = time.time()
    candidate_recall_hits = 0
    total_true_links = sum(len(ms) for ms in val_gt.values())

    # Collect ALL scored pairs first so we can apply global bipartite matching,
    # exactly like entity_resolution.py does at inference time.
    all_scored_pairs: list[tuple[float, str, str]] = []
    queries_with_no_candidates: set[str] = set()

    for eid, qn, qa, q_nums in queries:
        truth = val_gt[eid]
        query_keys = generate_blocking_keys(qn, qa)
        candidates = index.get_candidates(query_keys, max_key_frequency=MAX_KEY_FREQUENCY, top_k=TOP_K)
        if not candidates:
            queries_with_no_candidates.add(eid)
            continue

        cand_eids = [target_ids[t_idx] for t_idx, _ in candidates]
        candidate_recall_hits += len(truth & set(cand_eids))

        features_batch = []
        for t_idx, block_weight in candidates:
            tn, ta, t_nums = target_records[t_idx]
            feat = extract_pair_features(qn, qa, q_nums, tn, ta, t_nums, block_weight)
            features_batch.append(feat)

        probs = booster.predict(features_batch)
        for i, p in enumerate(probs):
            if p >= threshold:
                all_scored_pairs.append((float(p), eid, cand_eids[i]))

    t1 = time.time()
    print(f"Inference complete in {t1 - t0:.2f}s ({len(queries) / (t1 - t0):.1f} queries/sec).", flush=True)
    print(f"Candidate Recall on held-out split: {candidate_recall_hits:,} / {total_true_links:,} "
          f"({candidate_recall_hits / max(total_true_links, 1) * 100:.2f}%)", flush=True)

    # Global competitive bipartite matching — mirrors entity_resolution.py exactly.
    print("Applying global bipartite matching (matches production inference)...", flush=True)
    all_scored_pairs.sort(key=lambda x: x[0], reverse=True)
    claimed_targets: set[str] = set()
    predictions: dict[str, set[str]] = defaultdict(set)

    for p, eid, cand in all_scored_pairs:
        if cand in claimed_targets:
            continue
        if len(predictions[eid]) >= MATCH_CAP:
            continue
        predictions[eid].add(cand)
        claimed_targets.add(cand)

    # Compute exact Macro F0.5
    total_f05 = 0.0
    for sid, truth in val_gt.items():
        pred = predictions.get(sid, set())
        if not truth:
            total_f05 += 1.0 if not pred else 0.0
            continue
        tp = len(pred & truth)
        if not tp:
            continue
        precision = tp / len(pred)
        recall = tp / len(truth)
        total_f05 += (1.25 * precision * recall) / (0.25 * precision + recall)

    macro_f05 = total_f05 / len(val_gt)
    print(f"\n=======================================================", flush=True)
    print(f">>> Held-out Validation Macro F_0.5 Score: {macro_f05:.4f} <<<", flush=True)
    print(f">>> (top_k={TOP_K}, max_key_freq={MAX_KEY_FREQUENCY}, threshold={threshold}, "
          f"match_cap={MATCH_CAP}, distractor_cap={distractor_cap:,}) <<<", flush=True)
    print(f"=======================================================\n", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-dir", type=Path,
        default=Path("/Users/gaurangbhatia/Projects/ml-challenge/dataset/train"),
    )
    parser.add_argument("--model-path", type=Path, default=Path(__file__).parent / "model.txt")
    parser.add_argument(
        "--val-holdout", type=int, default=15000,
        help="Must match train_model.py's --val-holdout for the split to be disjoint.",
    )
    parser.add_argument("--threshold", type=float, default=0.60)  # was 0.74 — now matches production
    parser.add_argument("--distractor-cap", type=int, default=1_000_000)
    args = parser.parse_args()
    evaluate(args.train_dir, args.model_path, args.val_holdout, args.threshold, args.distractor_cap)