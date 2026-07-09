#!/bin/bash
# Train DroNet (supervised steering head) on the IDSIA Forest Trails dataset.
#
# Usage:
#   bash sims/training/train_dronet.sh                  # defaults (30 epochs, small model)
#   bash sims/training/train_dronet.sh --epochs 50      # pass extra args through
#
# Prerequisites:
#   1. Place the IDSIA files-archive zip at datasets/idsia/files-archive
#   2. Run: python sims/training/extract_idsia.py
#   3. conda activate xpurt  (or any env with torch + torchvision)
#
# Output: logs/dronet/<timestamp>/{best.pt, last.pt, config.json, history.json}

set -euo pipefail
cd "$(dirname "$0")/../.."

echo "Training DroNet on IDSIA Forest Trails dataset..."
echo "Logs will be saved to: logs/dronet/"
echo ""

python -u sims/training/train_dronet.py "$@"
