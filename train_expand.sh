#!/bin/bash

# EXPANDING VOCAB OVER TIME
# Each run uses BIGGER vocab size on SAME corpus

MODEL="model.json"
CORPUS="massive_corpus.txt"

# Read current size from model if exists
if [ -f "$MODEL" ]; then
    CURRENT=$(python3 -c "import json; m=json.load(open('$MODEL')); print(len(m['vocab']))")
    echo "Current vocab: $CURRENT"
else
    CURRENT=0
    echo "No existing model"
fi

# Expand by 500 each run
NEW_SIZE=$((CURRENT + 500))
echo "Training to $NEW_SIZE vocab..."

# Retrain on full corpus with bigger target
cargo run -- --corpus "$CORPUS" --save "$MODEL" --vocab-size "$NEW_SIZE" --text "test"

# Verify
python3 -c "import json; m=json.load(open('$MODEL')); print(f'New vocab: {len(m[\\\"vocab\\\"])} | Merges: {len(m[\\\"merges\\\"])}')"
