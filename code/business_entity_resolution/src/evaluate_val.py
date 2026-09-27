"""Evaluates Macro F0.5 on held-out training data using the trained LightGBM model."""
from __future__ import annotations

import csv
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


def evaluate(train_dir: Path, model_path: Path, eval_sample: int = 15000, threshold: float = 0.60) -> None:
    print(f"Loading LightGBM model from {model_path}...", flush=True)
    booster = lgb.Booster(model_file=str(model_path))

    print(f"Loading ground truth sample ({eval_sample:,} queries)...", flush=True)
    # Use queries starting after 35,000 to ensure completely held-out evaluation
    val_gt: dict[str, set[str]] = {}
    with open(train_dir / "train_ground_truth.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i < 35000:
                continue
            ms = [m for m in row["matched_entity_ids"].split(",") if m]
            val_gt[row["source1_entity_id"]] = set(ms)
            if len(val_gt) >= eval_sample:
                break

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

    print("Building target index with 150,000 distractors...", flush=True)
    target_ids: list[str] = []
    target_records: list[tuple[str, str, set[str]]] = []
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
                    target_records.append((tn, ta, set(t_nums)))
                    keys = generate_blocking_keys(row["business_name"], row["business_address"])
                    index.add(t_idx, keys)

    print(f"Total targets indexed: {len(target_ids):,}. Running inference...", flush=True)
    t0 = time.time()
    predictions: dict[str, set[str]] = {}
    candidate_recall_hits = 0
    total_true_links = sum(len(ms) for ms in val_gt.values())

    for eid, qn, qa, q_nums in queries:
        truth = val_gt[eid]
        query_keys = generate_blocking_keys(qn, qa)
        for num in q_nums:
            if len(num) >= 2:
                query_keys.add(f"NUM|{num}")
        candidates = index.get_candidates(query_keys, max_key_frequency=1500, top_k=60)
        if not candidates:
            predictions[eid] = set()
            continue

        cand_eids = [target_ids[t_idx] for t_idx, _ in candidates]
        candidate_recall_hits += len(truth & set(cand_eids))

        features_batch = []
        for t_idx, block_weight in candidates:
            tn, ta, t_nums = target_records[t_idx]
            feat = extract_pair_features(qn, qa, q_nums, tn, ta, t_nums, block_weight)
            features_batch.append(feat)

        probs = booster.predict(features_batch)
        matched_candidates = []
        for i, p in enumerate(probs):
            if p < threshold:
                continue
            matched_candidates.append((cand_eids[i], p))
        matched_candidates.sort(key=lambda x: x[1], reverse=True)
        matched = {m[0] for m in matched_candidates[:12]}
        predictions[eid] = matched

    t1 = time.time()
    print(f"Inference complete in {t1 - t0:.2f}s ({len(queries) / (t1 - t0):.1f} queries/sec).", flush=True)
    print(f"Candidate Recall on held-out split: {candidate_recall_hits:,} / {total_true_links:,} ({candidate_recall_hits / total_true_links * 100:.2f}%)", flush=True)

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
    print(f"=======================================================\n", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("/Users/gaurangbhatia/Projects/ml-challenge/dataset/train"))
    parser.add_argument("--model-path", type=Path, default=Path(__file__).parent / "model.txt")
    parser.add_argument("--sample", type=int, default=15000)
    parser.add_argument("--threshold", type=float, default=0.74)
    args = parser.parse_args()
    evaluate(args.train_dir, args.model_path, args.sample, args.threshold)
