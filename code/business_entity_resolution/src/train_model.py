"""Trains a high-precision LightGBM re-ranking model with hard negative mining."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
from normalizer import clean_business_name, clean_address
from blocking import generate_blocking_keys, InvertedIndex
from features import extract_pair_features, FEATURE_NAMES

csv.field_size_limit(sys.maxsize)


def train(train_dir: Path, output_model_path: Path, max_queries: int = 40000) -> None:
    random.seed(42)
    np.random.seed(42)

    print(f"Reading ground truth from {train_dir / 'train_ground_truth.tsv'}...", flush=True)
    gt_map: dict[str, set[str]] = {}
    with open(train_dir / "train_ground_truth.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            matches = [m for m in row["matched_entity_ids"].split(",") if m]
            gt_map[row["source1_entity_id"]] = set(matches)
            if len(gt_map) >= max_queries:
                break

    needed_s1 = set(gt_map.keys())
    needed_targets = {m for ms in gt_map.values() for m in ms}
    singletons_in_sample = sum(1 for ms in gt_map.values() if not ms)
    print(
        f"Sampled {len(needed_s1):,} S1 queries ({singletons_in_sample:,} singletons) "
        f"with {len(needed_targets):,} true target matches.",
        flush=True,
    )

    s1_records: dict[str, tuple[str, str, set[str], str, str]] = {}
    with open(train_dir / "train_source1.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            eid = row["entity_id"]
            if eid in needed_s1:
                raw_n = row["business_name"]
                raw_a = row["business_address"]
                qn, _ = clean_business_name(raw_n)
                qa, _, q_nums = clean_address(raw_a)
                s1_records[eid] = (qn, qa, set(q_nums), raw_n, raw_a)
            if len(s1_records) == len(needed_s1):
                break

    print("Loading target records and building blocking index...", flush=True)
    target_ids: list[str] = []
    target_data: list[tuple[str, str, set[str]]] = []
    index = InvertedIndex()

    distractor_cap = 150000
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

    # 1. Natural candidate pairs from the inverted index
    for sid, (qn, qa, q_nums, raw_n, raw_a) in s1_records.items():
        truth_set = gt_map[sid]
        query_keys = generate_blocking_keys(raw_n, raw_a)
        candidates = index.get_candidates(query_keys, max_key_frequency=1200, top_k=45)
        for t_idx, block_weight in candidates:
            eid = target_ids[t_idx]
            tn, ta, t_nums = target_data[t_idx]
            feat = extract_pair_features(qn, qa, q_nums, tn, ta, t_nums, block_weight)
            X.append(feat)
            y.append(1 if eid in truth_set else 0)

    # 2. Perfect self-match positive anchors: teaches model that identical names & addresses are 100% positive
    print("Adding identity positive anchors...", flush=True)
    for sid, (qn, qa, q_nums, _, _) in s1_records.items():
        if gt_map[sid]:  # non-singleton
            feat = extract_pair_features(qn, qa, q_nums, qn, qa, q_nums, block_weight=8)
            X.append(feat)
            y.append(1)

    # 3. Hard spatial negatives: different business at the SAME address
    # Teaches the model that identical address with different name (name_ratio < 0.30) is NEGATIVE
    print("Mining hard spatial negatives (same address, different business)...", flush=True)
    num_spatial_negs = 0
    target_indices = list(range(len(target_data)))
    for sid, (qn, qa, q_nums, _, _) in s1_records.items():
        if not qa or len(qa.split()) < 3:
            continue
        for _ in range(2):
            rnd_idx = random.choice(target_indices)
            tn_diff, _, _ = target_data[rnd_idx]
            if not tn_diff or fuzz.ratio(qn, tn_diff) >= 30:
                continue
            # Pair: query address + distinct target name
            feat = extract_pair_features(qn, qa, q_nums, tn_diff, qa, q_nums, block_weight=8)
            X.append(feat)
            y.append(0)
            num_spatial_negs += 1
        if num_spatial_negs >= 25000:
            break

    # 4. Hard multi-location negatives: same name at a completely different address
    # Teaches the model that generic business names on different streets/cities are NEGATIVE
    print("Mining hard name-match negatives (same name, different address)...", flush=True)
    num_name_negs = 0
    for sid, (qn, qa, q_nums, _, _) in s1_records.items():
        if not qn or len(qn.split()) < 2:
            continue
        for _ in range(2):
            rnd_idx = random.choice(target_indices)
            _, ta_diff, t_nums_diff = target_data[rnd_idx]
            if not ta_diff or fuzz.ratio(qa, ta_diff) >= 30 or len(q_nums & t_nums_diff) > 0:
                continue
            feat = extract_pair_features(qn, qa, q_nums, qn, ta_diff, t_nums_diff, block_weight=8)
            X.append(feat)
            y.append(0)
            num_name_negs += 1
        if num_name_negs >= 25000:
            break

    pos_count = sum(y)
    neg_count = len(y) - pos_count
    print(
        f"Training dataset: {len(X):,} pairs ({pos_count:,} positive, {neg_count:,} negative).",
        flush=True,
    )

    print("Fitting LightGBM model...", flush=True)
    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.07,
        num_leaves=40,
        max_depth=8,
        min_child_samples=30,
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
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent / "dataset" / "train",
    )
    parser.add_argument("--output-model", type=Path, default=Path(__file__).parent / "model.txt")
    parser.add_argument("--max-queries", type=int, default=40000)
    args = parser.parse_args()
    train(args.train_dir, args.output_model, args.max_queries)
