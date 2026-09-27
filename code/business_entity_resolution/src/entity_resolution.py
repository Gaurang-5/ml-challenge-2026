"""
End-to-End Business Entity Resolution Inference Pipeline.

Features:
- Multi-Key Inverted Index with lean candidate generation (default: top_k = 20)
- Dual backend: PyTorch Deep Learning (Apple GPU/MPS & CUDA) or LightGBM GBDT
- Global Competitive Bipartite Matching enforcing 1-to-1 Target Invariant
- Scalable, country-isolated execution under 4GB RAM
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
import time

import numpy as np
import torch

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

sys.path.insert(0, str(Path(__file__).parent))
from normalizer import clean_business_name, clean_address
from blocking import generate_blocking_keys, InvertedIndex
from features import extract_pair_features
from neural_model import EntityResolutionNet

csv.field_size_limit(sys.maxsize)


def run_pipeline(
    test_dir: Path,
    output_dir: Path,
    model_path: Path | None = None,
    threshold: float = 0.60,
    top_candidates: int = 20,
    max_key_freq: int = 1500,
) -> None:
    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pytorch_model = None
    booster = None

    # Load model
    if model_path and model_path.exists():
        if model_path.suffix == ".pt":
            print(f"Loading PyTorch Model from {model_path} onto {device}...", flush=True)
            ckpt = torch.load(str(model_path), map_location=device)
            pytorch_model = EntityResolutionNet(
                input_dim=ckpt.get("input_dim", 15),
                hidden_dims=ckpt.get("hidden_dims", [128, 64, 32]),
            ).to(device)
            pytorch_model.load_state_dict(ckpt["model_state_dict"])
            pytorch_model.eval()
        elif lgb is not None:
            print(f"Loading LightGBM model from {model_path}...", flush=True)
            booster = lgb.Booster(model_file=str(model_path))
    else:
        print("Warning: No model file found, falling back to heuristic scoring.", flush=True)

    # Pass 1: Scan Source 1 queries preserving exact file order
    s1_path = test_dir / "test_source1.tsv"
    print(f"Scanning Source 1 queries from {s1_path}...", flush=True)

    country_queries: dict[str, list[tuple[str, str, str, set[str], str, str]]] = defaultdict(list)
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

            country_queries[country].append((eid, qn, qa, set(q_nums), raw_n, raw_a))

    print(
        f"Loaded {len(ordered_s1_ids):,} Source 1 entities across countries: {list(country_queries.keys())}",
        flush=True,
    )

    results_matching: dict[str, str] = {}
    results_candidates: dict[str, str] = {}

    f_match = open(matching_file, "w", encoding="utf-8", newline="")
    f_cand = open(candidate_file, "w", encoding="utf-8", newline="")
    m_writer = csv.writer(f_match, delimiter="\t")
    c_writer = csv.writer(f_cand, delimiter="\t")
    m_writer.writerow(["source1_entity_id", "matched_entity_ids"])
    c_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

    # Pass 2: Process each country partition
    for country, queries in country_queries.items():
        print(f"\n=======================================================")
        print(f"Processing Country: [{country}] ({len(queries):,} queries)")
        print(f"=======================================================")

        # Build in-memory target index
        print(f"Building in-memory inverted index for [{country}] targets...", flush=True)
        index = InvertedIndex()
        target_ids: list[str] = []
        target_records: list[tuple[str, str, set[str]]] = []

        for target_file in ["test_source2.tsv", "test_source3.tsv"]:
            t_path = test_dir / target_file
            if not t_path.exists():
                continue
            with open(t_path, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    c = (row.get("country") or "__unknown__").strip()
                    if c != country:
                        continue
                    t_idx = len(target_ids)
                    target_ids.append(row["entity_id"])
                    tn, _ = clean_business_name(row["business_name"])
                    ta, _, t_nums = clean_address(row["business_address"])
                    target_records.append((tn, ta, set(t_nums)))
                    keys = generate_blocking_keys(row["business_name"], row["business_address"])
                    index.add(t_idx, keys)

        t_country_start = time.time()
        print(f"Indexed {len(target_ids):,} targets for [{country}]. Scoring queries...", flush=True)

        batch_size = 4000
        country_scored_pairs: list[tuple[float, str, str]] = []

        for b_start in range(0, len(queries), batch_size):
            batch_queries = queries[b_start : b_start + batch_size]
            batch_features: list[list[float]] = []
            query_slices: list[tuple[str, list[str], int, int]] = []

            for eid, qn, qa, q_nums, raw_n, raw_a in batch_queries:
                query_keys = generate_blocking_keys(raw_n, raw_a)
                candidates = index.get_candidates(query_keys, max_key_frequency=max_key_freq, top_k=top_candidates)

                if not candidates:
                    results_candidates[eid] = ""
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

            # Model inference
            if batch_features:
                if pytorch_model is not None:
                    with torch.no_grad():
                        t_feat = torch.tensor(batch_features, dtype=torch.float32, device=device)
                        probs = pytorch_model(t_feat).cpu().numpy()
                elif booster is not None:
                    probs = booster.predict(batch_features)
                else:
                    probs = [
                        (0.5 * f[0] + 0.4 * f[6] + 0.1 * f[10]) if f[5] == 0.0 else (f[0] * 0.95)
                        for f in batch_features
                    ]

                for eid, cand_eids, s_idx, e_idx in query_slices:
                    q_probs = probs[s_idx:e_idx]
                    for i, p in enumerate(q_probs):
                        if p >= threshold:
                            country_scored_pairs.append((float(p), eid, cand_eids[i]))

            processed = min(b_start + batch_size, len(queries))
            if processed % 50000 == 0 or processed == len(queries):
                elapsed = time.time() - t_country_start
                rate = processed / max(elapsed, 0.01)
                print(f"  [{country}] Scoring: {processed:,} / {len(queries):,} ({rate:.1f} queries/sec)", flush=True)

        print(f"Applying Competitive Bipartite Matching for [{country}] ({len(country_scored_pairs):,} candidates)...", flush=True)
        # Sort descending by probability: highest confidence matches claim targets first
        country_scored_pairs.sort(key=lambda x: x[0], reverse=True)
        claimed_targets = set()
        country_matches = defaultdict(list)

        for p, q_eid, c_eid in country_scored_pairs:
            if c_eid in claimed_targets:
                continue
            if len(country_matches[q_eid]) >= 10:
                continue
            country_matches[q_eid].append(c_eid)
            claimed_targets.add(c_eid)

        for q_eid, matches in country_matches.items():
            results_matching[q_eid] = ",".join(matches)

        print(f"Finished [{country}] in {time.time() - t_country_start:.2f}s (Resolved {len(claimed_targets):,} unique target matches).", flush=True)

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
    parser.add_argument("--model-path", type=Path, default=Path(__file__).parent / "model.pt")
    parser.add_argument("--threshold", type=float, default=0.60, help="Confidence threshold for final match")
    parser.add_argument("--top-candidates", type=int, default=20, help="Max candidates per query")
    parser.add_argument("--max-key-freq", type=int, default=1500, help="Max blocking key frequency")
    args = parser.parse_args()

    # Fallback to model.txt if model.pt doesn't exist yet
    model_path = args.model_path
    if not model_path.exists() and (model_path.parent / "model.txt").exists():
        model_path = model_path.parent / "model.txt"

    run_pipeline(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=model_path,
        threshold=args.threshold,
        top_candidates=args.top_candidates,
        max_key_freq=args.max_key_freq,
    )


if __name__ == "__main__":
    main()
