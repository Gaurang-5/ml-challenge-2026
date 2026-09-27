"""
End-to-End Business Entity Resolution Inference Pipeline.

Features:
- Multi-Key Inverted Index with lean candidate generation (default: top_k = 60,
  matching train_model.py / evaluate_val.py exactly)
- LightGBM GBDT re-ranking (canonical model for this submission)
- Global Competitive Bipartite Matching enforcing 1-to-1 Target Invariant
- Scalable, country-isolated execution
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
import time

import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import torch
    from neural_model import EntityResolutionNet
except ImportError:
    torch = None
    EntityResolutionNet = None

sys.path.insert(0, str(Path(__file__).parent))
from normalizer import clean_business_name, clean_address, normalize_country
from blocking import generate_blocking_keys, InvertedIndex
from features import extract_pair_features

csv.field_size_limit(sys.maxsize)

# Canonical blocking / matching parameters — MUST match train_model.py and evaluate_val.py.
DEFAULT_TOP_K = 60
DEFAULT_MAX_KEY_FREQ = 1500
DEFAULT_MATCH_CAP = 12


def load_model(model_path: Path):
    """Loads exactly one model type and states unambiguously which one, so the
    methodology doc and the running code can never silently diverge."""
    if model_path.suffix == ".txt":
        if lgb is None:
            raise RuntimeError("lightgbm is not installed but a .txt (LightGBM) model was requested.")
        print(f"[MODEL] Loading LightGBM booster from {model_path}", flush=True)
        return "lightgbm", lgb.Booster(model_file=str(model_path)), None

    if model_path.suffix == ".pt":
        if torch is None or EntityResolutionNet is None:
            raise RuntimeError("torch is not installed but a .pt (PyTorch) model was requested.")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[MODEL] Loading PyTorch model from {model_path} onto {device}", flush=True)
        ckpt = torch.load(str(model_path), map_location=device)
        model = EntityResolutionNet(
            input_dim=ckpt.get("input_dim", 15),
            hidden_dims=ckpt.get("hidden_dims", [128, 64, 32]),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return "pytorch", model, device

    raise ValueError(f"Unrecognized model file extension: {model_path.suffix}")


def run_pipeline(
    test_dir: Path,
    output_dir: Path,
    model_path: Path,
    threshold: float = 0.60,
    top_candidates: int = DEFAULT_TOP_K,
    max_key_freq: int = DEFAULT_MAX_KEY_FREQ,
    match_cap: int = DEFAULT_MATCH_CAP,
) -> None:
    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"

    model_kind, model_obj, device = load_model(model_path)
    booster = model_obj if model_kind == "lightgbm" else None
    pytorch_model = model_obj if model_kind == "pytorch" else None

    print(f"[CONFIG] top_candidates={top_candidates}, max_key_freq={max_key_freq}, "
          f"threshold={threshold}, match_cap={match_cap}", flush=True)

    # Pass 1: Scan Source 1 queries preserving exact file order
    s1_path = test_dir / "test_source1.tsv"
    print(f"Scanning Source 1 queries from {s1_path}...", flush=True)

    country_queries: dict[str, list[tuple[str, str, str, set[str], str, str]]] = defaultdict(list)
    ordered_s1_ids: list[str] = []

    with open(s1_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row["entity_id"]
            country = normalize_country(row.get("country"))
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
                    c = normalize_country(row.get("country"))
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

            if batch_features:
                if pytorch_model is not None:
                    with torch.no_grad():
                        t_feat = torch.tensor(batch_features, dtype=torch.float32, device=device)
                        probs = pytorch_model(t_feat).cpu().numpy()
                elif booster is not None:
                    probs = booster.predict(batch_features)
                else:
                    raise RuntimeError("No model loaded — this should be unreachable given load_model().")

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
        country_scored_pairs.sort(key=lambda x: x[0], reverse=True)
        claimed_targets = set()
        country_matches = defaultdict(list)

        for p, q_eid, c_eid in country_scored_pairs:
            if c_eid in claimed_targets:
                continue
            if len(country_matches[q_eid]) >= match_cap:
                continue
            country_matches[q_eid].append(c_eid)
            claimed_targets.add(c_eid)

        for q_eid, matches in country_matches.items():
            results_matching[q_eid] = ",".join(matches)

        print(f"Finished [{country}] in {time.time() - t_country_start:.2f}s "
              f"(Resolved {len(claimed_targets):,} unique target matches).", flush=True)

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
    parser.add_argument(
        "--model-path", type=Path, default=Path(__file__).parent / "model.txt",
        help="Explicit path. Defaults to the LightGBM model.txt — the canonical model "
             "described in the methodology doc. Pass a .pt path explicitly to use PyTorch instead.",
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--top-candidates", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-key-freq", type=int, default=DEFAULT_MAX_KEY_FREQ)
    parser.add_argument("--match-cap", type=int, default=DEFAULT_MATCH_CAP)
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {args.model_path}. "
            f"Pass --model-path explicitly if it's not at the default location."
        )

    run_pipeline(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        top_candidates=args.top_candidates,
        max_key_freq=args.max_key_freq,
        match_cap=args.match_cap,
    )


if __name__ == "__main__":
    main()