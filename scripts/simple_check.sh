#!/bin/bash
echo "Validating data files..."
if [ -f "/a0/bitcore/workspace/data/sequences/original/GDPa1_v1.2_sequences.csv" ]; then
  echo "✓ GDPa1 file exists"
else
  echo "✗ GDPa1 file missing"
  exit 1
fi
if [ -f "/a0/bitcore/workspace/data/sequences/heldout-set-sequences.csv" ]; then
  echo "✓ Heldout file exists"
else
  echo "✗ Heldout file missing"
  exit 1
fi
echo "Validation complete. Ready to proceed."
