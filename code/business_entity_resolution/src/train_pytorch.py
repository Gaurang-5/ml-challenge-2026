#!/usr/bin/env python3
"""
PyTorch Deep Learning Training for Business Entity Resolution.

Trains an EntityResolutionNet using AdamW, Cosine Annealing, and MacroF05Loss.
Streams live training progress and validation metrics directly to stdout.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
from normalizer import clean_business_name, clean_address
from blocking import generate_blocking_keys, InvertedIndex
from features import extract_pair_features
from neural_model import EntityResolutionNet, MacroF05Loss

csv.field_size_limit(sys.maxsize)


def train_pytorch(
    train_dir: Path,
    output_model_path: Path,
    max_queries: int = 50000,
    epochs: int = 15,
    batch_size: int = 1024,
    lr: float = 3e-3,
) -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=======================================================")
    print(f"🔥 PyTorch Deep Learning Entity Resolution Trainer")
    print(f"Compute Device: {device} (Apple Silicon GPU Acceleration)" if device.type == "mps" else f"Compute Device: {device}")
    print(f"=======================================================\n")

    print(f"Loading training data from {train_dir}...", flush=True)
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
    singletons = sum(1 for ms in gt_map.values() if not ms)
    print(f"Sampled {len(needed_s1):,} S1 queries ({singletons:,} singletons, {len(needed_targets):,} true target matches).", flush=True)

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

    print("Indexing target records and building blocking candidates...", flush=True)
    target_ids: list[str] = []
    target_data: list[tuple[str, str, set[str]]] = []
    index = InvertedIndex()

    distractor_cap = 200000
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

    print(f"Indexed {len(target_ids):,} targets. Mining training pairs...", flush=True)
    X: list[list[float]] = []
    y: list[float] = []

    # 1. Blocking candidates
    for sid, (qn, qa, q_nums, raw_n, raw_a) in s1_records.items():
        truth_set = gt_map[sid]
        query_keys = generate_blocking_keys(raw_n, raw_a)
        candidates = index.get_candidates(query_keys, max_key_frequency=1200, top_k=25)
        for t_idx, block_weight in candidates:
            eid = target_ids[t_idx]
            tn, ta, t_nums = target_data[t_idx]
            feat = extract_pair_features(qn, qa, q_nums, tn, ta, t_nums, block_weight)
            X.append(feat)
            y.append(1.0 if eid in truth_set else 0.0)

    # 2. Positive identity anchors
    for sid, (qn, qa, q_nums, _, _) in s1_records.items():
        if gt_map[sid]:
            feat = extract_pair_features(qn, qa, q_nums, qn, qa, q_nums, block_weight=8)
            X.append(feat)
            y.append(1.0)

    # 3. Hard spatial negatives (same address, different business name)
    target_indices = list(range(len(target_data)))
    num_spatial = 0
    for sid, (qn, qa, q_nums, _, _) in s1_records.items():
        if not qa or len(qa.split()) < 3:
            continue
        for _ in range(2):
            rnd_idx = random.choice(target_indices)
            tn_diff, _, _ = target_data[rnd_idx]
            if not tn_diff or fuzz.ratio(qn, tn_diff) >= 30:
                continue
            feat = extract_pair_features(qn, qa, q_nums, tn_diff, qa, q_nums, block_weight=8)
            X.append(feat)
            y.append(0.0)
            num_spatial += 1
        if num_spatial >= 30000:
            break

    # 4. Hard name negatives (same business name, different location)
    num_name = 0
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
            y.append(0.0)
            num_name += 1
        if num_name >= 30000:
            break

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    pos_count = int(y_arr.sum())
    neg_count = len(y_arr) - pos_count
    print(f"Total training pairs: {len(X_arr):,} ({pos_count:,} positive, {neg_count:,} negative).", flush=True)

    # Split train and validation sets
    indices = np.random.permutation(len(X_arr))
    split = int(0.90 * len(X_arr))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train_t = torch.tensor(X_arr[train_idx])
    y_train_t = torch.tensor(y_arr[train_idx])
    X_val_t = torch.tensor(X_arr[val_idx])
    y_val_t = torch.tensor(y_arr[val_idx])

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size * 2, shuffle=False)

    model = EntityResolutionNet(input_dim=15, hidden_dims=[128, 64, 32], dropout=0.2).to(device)
    criterion = MacroF05Loss(alpha=0.35, gamma=2.0, fp_weight=2.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n🚀 Starting Neural Network Training on {device} ({epochs} Epochs)...")
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Val Loss':<14}{'Val Precision':<16}{'Val Recall':<14}{'Val F0.5':<12}{'Time':<8}")
    print("-" * 86)

    best_val_f05 = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(by)

        scheduler.step()
        train_loss = total_train_loss / len(train_idx)

        # Validation evaluation
        model.eval()
        total_val_loss = 0.0
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx)
                loss = criterion(preds, by)
                total_val_loss += loss.item() * len(by)
                val_preds_list.append(preds.cpu())
                val_targets_list.append(by.cpu())

        val_loss = total_val_loss / len(val_idx)
        all_preds = torch.cat(val_preds_list).numpy()
        all_targets = torch.cat(val_targets_list).numpy()

        binary_preds = (all_preds >= 0.50).astype(float)
        tp = np.sum((binary_preds == 1.0) & (all_targets == 1.0))
        fp = np.sum((binary_preds == 1.0) & (all_targets == 0.0))
        fn = np.sum((binary_preds == 0.0) & (all_targets == 1.0))

        prec = tp / max(tp + fp, 1e-6)
        rec = tp / max(tp + fn, 1e-6)
        f05 = (1.25 * prec * rec) / max(0.25 * prec + rec, 1e-6)

        elapsed = time.time() - t0
        print(
            f"{epoch:<8}{train_loss:<14.4f}{val_loss:<14.4f}{prec:<16.4f}{rec:<14.4f}{f05:<12.4f}{elapsed:<6.1f}s",
            flush=True,
        )

        if f05 > best_val_f05:
            best_val_f05 = f05
            output_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": 15,
                "hidden_dims": [128, 64, 32],
                "val_f05": best_val_f05,
                "epoch": epoch,
            }, str(output_model_path))

    print("-" * 86)
    print(f"🎉 Training Complete! Best Validation F0.5: {best_val_f05:.4f}")
    print(f"PyTorch model saved to: {output_model_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path(__file__).parent.parent.parent.parent / "dataset" / "train")
    parser.add_argument("--output-model", type=Path, default=Path(__file__).parent / "model.pt")
    parser.add_argument("--max-queries", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    train_pytorch(
        train_dir=args.train_dir,
        output_model_path=args.output_model,
        max_queries=args.max_queries,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
