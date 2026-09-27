#!/usr/bin/env python3
"""High-precision, high-recall business entity resolution pipeline.

Orchestrates multi-key blocking, C++ RapidFuzz feature extraction,
and vectorized LightGBM re-ranking across country partitions.
"""
from __future__ import annotations

import argparse
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


def run_pipeline(
    test_dir: Path,
    output_dir: Path,
    model_path: Path | None = None,
    threshold: float = 0.60,
    top_candidates: int = 60,
    max_key_freq: int = 1500,
) -> None:
    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"

    # Load trained model if available
    booster = None
    if model_path and model_path.exists():
        print(f"Loading LightGBM model from {model_path}...", flush=True)
        booster = lgb.Booster(model_file=str(model_path))
    else:
        print("Warning: No model file found, falling back to rule-based heuristic scoring.", flush=True)

    # Pass 1: Read all test Source 1 records to discover countries and preserve original order
    s1_path = test_dir / "test_source1.tsv"
    print(f"Scanning Source 1 queries from {s1_path}...", flush=True)
    
    queries_by_country: dict[str, list[tuple[str, str, str, set[str]]]] = {}
    ordered_s1_ids: list[str] = []
    
    with open(s1_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row["entity_id"]
            country = (row.get("country") or "__unknown__").strip()
            ordered_s1_ids.append(eid)
            
            raw_n = row["business_name"]
            raw_a = row["business_address"]
            qn, _ = clean_business_name(raw_n)
            qa, _, q_nums = clean_address(raw_a)
            
            if country not in queries_by_country:
                queries_by_country[country] = []
            queries_by_country[country].append((eid, qn, qa, set(q_nums), raw_n, raw_a))

    total_s1 = len(ordered_s1_ids)
    print(f"Loaded {total_s1:,} Source 1 entities across countries: {list(queries_by_country.keys())}", flush=True)

    # Open output writers immediately to stream results to disk and keep memory minimal
    f_match = open(matching_file, "w", encoding="utf-8", newline="")
    f_cand = open(candidate_file, "w", encoding="utf-8", newline="")
    m_writer = csv.writer(f_match, delimiter="\t", lineterminator="\n")
    c_writer = csv.writer(f_cand, delimiter="\t", lineterminator="\n")
    m_writer.writerow(["source1_entity_id", "matched_entity_ids"])
    c_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

    results_matching: dict[str, str] = {}
    results_candidates: dict[str, str] = {}

    target_files = [test_dir / "test_source2.tsv", test_dir / "test_source3.tsv"]

    # Process each country independently to isolate memory and eliminate cross-country false positives
    for country, queries in queries_by_country.items():
        t_country_start = time.time()
        print(f"\n=======================================================", flush=True)
        print(f"Processing Country: [{country}] ({len(queries):,} queries)", flush=True)
        print(f"=======================================================", flush=True)

        target_ids: list[str] = []
        target_records: list[tuple[str, str, set[str]]] = []
        index = InvertedIndex()

        print(f"Building in-memory inverted index for [{country}] targets...", flush=True)
        for t_path in target_files:
            if not t_path.exists():
                print(f"Warning: {t_path} not found!", flush=True)
                continue
            with open(t_path, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    row_c = (row.get("country") or "__unknown__").strip()
                    if row_c != country:
                        continue
                    t_idx = len(target_ids)
                    eid = row["entity_id"]
                    target_ids.append(eid)
                    
                    tn, _ = clean_business_name(row["business_name"])
                    ta, _, t_nums = clean_address(row["business_address"])
                    target_records.append((tn, ta, set(t_nums)))
                    
                    keys = generate_blocking_keys(row["business_name"], row["business_address"])
                    index.add(t_idx, keys)

        print(f"Indexed {len(target_ids):,} targets for [{country}]. Scoring queries...", flush=True)

        # Batch scoring queries to vectorize model inference
        batch_size = 2000
        for b_start in range(0, len(queries), batch_size):
            batch_queries = queries[b_start : b_start + batch_size]
            
            batch_features: list[list[float]] = []
            query_slices: list[tuple[str, list[str], int, int]] = []
            
            for eid, qn, qa, q_nums, raw_n, raw_a in batch_queries:
                query_keys = generate_blocking_keys(raw_n, raw_a)
                candidates = index.get_candidates(query_keys, max_key_frequency=max_key_freq, top_k=top_candidates)
                
                if not candidates:
                    results_candidates[eid] = ""
                    results_matching[eid] = ""
                    continue

                cand_eids = [target_ids[t_idx] for t_idx, _ in candidates]
                results_candidates[eid] = ",".join(cand_eids)

                start_idx = len(batch_features)
                for t_idx, block_weight in candidates:
                    tn, ta, t_nums = target_records[t_idx]
                    feat = extract_pair_features(qn, qa, q_nums, tn, ta, t_nums, block_weight)
                    batch_features.append(feat)
                end_idx = len(batch_features)

                query_slices.append((eid, cand_eids, start_idx, end_idx))

            # Vectorized prediction across the batch
            if batch_features:
                if booster is not None:
                    probs = booster.predict(batch_features)
                else:
                    probs = [
                        (0.5 * f[0] + 0.4 * f[6] + 0.1 * f[10]) if f[5] == 0.0 else (f[0] * 0.95)
                        for f in batch_features
                    ]

                for eid, cand_eids, s_idx, e_idx in query_slices:
                    q_probs = probs[s_idx:e_idx]
                    matched_candidates = []
                    for i, p in enumerate(q_probs):
                        if p < threshold:
                            continue
                        matched_candidates.append((cand_eids[i], p))

                    # Sort matches by model probability descending and cap at 12
                    matched_candidates.sort(key=lambda x: x[1], reverse=True)
                    capped_matches = [m[0] for m in matched_candidates[:12]]
                    results_matching[eid] = ",".join(capped_matches)

            processed = min(b_start + batch_size, len(queries))
            if processed % 50000 == 0 or processed == len(queries):
                elapsed = time.time() - t_country_start
                rate = processed / max(elapsed, 0.01)
                print(f"  [{country}] Progress: {processed:,} / {len(queries):,} ({rate:.1f} queries/sec)", flush=True)

        print(f"Finished [{country}] in {time.time() - t_country_start:.2f}s.", flush=True)

    # Write output files in exact original Source 1 order
    print(f"\nWriting final outputs to {output_dir}...", flush=True)
    for eid in ordered_s1_ids:
        m_writer.writerow([eid, results_matching.get(eid, "")])
        c_writer.writerow([eid, results_candidates.get(eid, "")])

    f_match.close()
    f_cand.close()

    print(f"\nPipeline successfully completed in {time.time() - t_start:.2f}s!", flush=True)
    print(f"  Matching:  {matching_file} ({matching_file.stat().st_size:,} bytes)")
    print(f"  Candidate: {candidate_file} ({candidate_file.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", type=Path, default=Path(__file__).parent / "model.txt")
    parser.add_argument("--threshold", type=float, default=0.60, help="Confidence threshold for final match")
    parser.add_argument("--top-candidates", type=int, default=60, help="Max candidates per query")
    parser.add_argument("--max-key-freq", type=int, default=1500, help="Max blocking key frequency")
    args = parser.parse_args()

    run_pipeline(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        top_candidates=args.top_candidates,
        max_key_freq=args.max_key_freq,
    )


if __name__ == "__main__":
    main()
