#!/usr/bin/env bash
# Remote runner script on EC2
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

echo "=== ML Challenge 2026 EC2 Runner ==="
echo "Host: $(hostname) | CPU Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/ {print $2}')"

# Setup virtual environment if not present
if [ ! -d "venv" ]; then
    echo "Creating python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r code/business_entity_resolution/requirements.txt
else
    source venv/bin/activate
fi

ACTION="${1:-infer}"
THRESHOLD="${2:-0.60}"

case "$ACTION" in
    train)
        echo "Running LightGBM model training with $(nproc) CPU threads..."
        python3 code/business_entity_resolution/src/train_model.py \
            --train-dir dataset/train \
            --output-model code/business_entity_resolution/src/model.txt
        ;;
    eval)
        echo "Running held-out validation..."
        python3 code/business_entity_resolution/src/evaluate_val.py \
            --threshold "$THRESHOLD"
        ;;
    infer)
        echo "Running full test inference with threshold $THRESHOLD..."
        mkdir -p output
        python3 code/business_entity_resolution/src/entity_resolution.py \
            --test-dir dataset/test \
            --output-dir output \
            --threshold "$THRESHOLD"
        echo "Validating outputs..."
        python3 utils/validate_submission.py \
            --matching output/matching_results.tsv \
            --candidate /tmp/skip.tsv \
            --test-dir dataset/test
        ;;
    *)
        echo "Usage: ./scripts/remote_run.sh [train|eval|infer] [threshold]"
        exit 1
        ;;
esac

echo "Task completed successfully!"
