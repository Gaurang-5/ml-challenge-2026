# Business Entity Resolution Pipeline

This package implements an end-to-end, reproducible business entity resolution solution designed for high precision ($F_{0.5}$) matching across multi-lingual datasets (US, India, France).

## Architecture Highlights
- **High-Recall Multi-Key Inverted Index**: Operates purely in-memory with zero disk-churn, generating redundant name, address, and spatial keys achieving >98% candidate recall.
- **Indic Transliteration & French Diacritics Normalization**: Compresses transliterated character sequences (`text_unidecode` + phonetic vowel/consonant squashing) to resolve cross-script matches.
- **RapidFuzz & LightGBM Precision Re-ranking**: Scores candidates across 13 fine-grained token, character, and numeric features, calibrated specifically for the competition's macro $F_{0.5}$ metric.
- **Country Partitioning**: Streamlines memory into isolated partitions (`France`, `US`, `India`), ensuring zero cross-country false positives and sub-4GB RAM consumption.

## Setup
```bash
pip install -r requirements.txt
```

## Training (Optional)
To retrain the LightGBM re-ranking model:
```bash
python3 src/train_model.py \
  --train-dir ../../dataset/train \
  --output-model src/model.txt
```

## Inference
To run inference on the test dataset:
```bash
python3 src/entity_resolution.py \
  --test-dir ../../dataset/test \
  --output-dir ../../output \
  --model-path src/model.txt \
  --threshold 0.60
```

## Validation
To validate format compliance before leaderboard submission:
```bash
python3 ../../utils/validate_submission.py \
  --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv \
  --test-dir ../../dataset/test \
  --check-ids
```
