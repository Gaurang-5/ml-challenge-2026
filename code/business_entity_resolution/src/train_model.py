"""Trains a high-precision LightGBM re-ranking model on training ground truth."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from normalizer import clean_business_name, clean_address
from blocking import generate_blocking_keys, InvertedIndex
from features import extract_pair_features, FEATURE_NAMES

csv.field_size_limit(sys.maxsize)


def train(train_dir: Path, output_model_path: Path, max_queries: int = 35000) -> None:
    print(f"Reading ground truth from {train_dir / 'train_ground_truth.tsv'}...", flush=True)
    gt_map: dict[str, set[str]] = {}
    with open(train_dir / "train_ground_truth.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            matches = [m for m in row["matched_entity_ids"].split(",") if m]
            if matches:
                gt_map[row["source1_entity_id"]] = set(matches)
            if len(gt_map) >= max_queries:
                break

    needed_s1 = set(gt_map.keys())
    needed_targets = {m for ms in gt_map.values() for m in ms}
    print(f"Sampled {len(needed_s1):,} S1 queries with {len(needed_targets):,} true target matches.", flush=True)

    s1_records: dict[str, tuple[str, str, set[str]]] = {}
    with open(train_dir / "train_source1.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            eid = row["entity_id"]
            if eid in needed_s1:
                qn, _ = clean_business_name(row["business_name"])
                qa, _, q_nums = clean_address(row["business_address"])
                s1_records[eid] = (qn, qa, set(q_nums))
            if len(s1_records) == len(needed_s1):
                break

    print("Loading target records and building blocking index...", flush=True)
    target_ids: list[str] = []
    target_data: list[tuple[str, str, set[str]]] = []
    index = InvertedIndex()
    
    # Load all needed true targets plus distractor targets
    distractor_cap = 100000
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
                    target_data.append((tn, ta, set(t_nums)))
                    keys = generate_blocking_keys(row["business_name"], row["business_address"])
                    index.add(t_idx, keys)

    print(f"Total targets indexed: {len(target_ids):,}. Generating training pairs...", flush=True)

    X: list[list[float]] = []
    y: list[int] = []

    for sid, (qn, qa, q_nums) in s1_records.items():
        truth_set = gt_map[sid]
        # Generate blocking keys from raw inputs or tokens
        # Reconstruct representative keys
        query_keys = generate_blocking_keys(qn, qa)
        for num in q_nums:
            if len(num) >= 2:
                query_keys.add(f"NUM|{num}")
        candidates = index.get_candidates(query_keys, max_key_frequency=1200, top_k=50)
        for t_idx, block_weight in candidates:
            eid = target_ids[t_idx]
            tn, ta, t_nums = target_data[t_idx]
            feat = extract_pair_features(qn, qa, q_nums, tn, ta, t_nums, block_weight)
            X.append(feat)
            y.append(1 if eid in truth_set else 0)

    pos_count = sum(y)
    neg_count = len(y) - pos_count
    print(f"Training dataset: {len(X):,} pairs ({pos_count:,} positive, {neg_count:,} negative).", flush=True)

    print("Fitting LightGBM model...", flush=True)
    clf = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.07,
        num_leaves=35,
        max_depth=7,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1,
    )
    clf.fit(X, y)

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    clf.booster_.save_model(str(output_model_path))
    print(f"Model successfully saved to {output_model_path}!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("/Users/gaurangbhatia/Projects/ml-challenge/dataset/train"))
    parser.add_argument("--output-model", type=Path, default=Path(__file__).parent / "model.txt")
    parser.add_argument("--max-queries", type=int, default=35000)
    args = parser.parse_args()
    train(args.train_dir, args.output_model, args.max_queries)
