#!/bin/bash

echo "=== Antibody Developability Project Workspace Validation ==="
echo "Date: $(date)"
echo ""

# Define known good hashes
GDPa1_HASH="9f3c68431802185e072f86e06ba65475fb6e4b097b5ff689486dd8ad266164f1"
HELDOUT_HASH="cab3383787b88808be863dc2a329f73a643d47e39f01fa6e83d372071a780609"

# Define file paths
GDPa1_FILE="/a0/bitcore/workspace/data/sequences/original/GDPa1_v1.2_sequences.csv"
HELDOUT_FILE="/a0/bitcore/workspace/data/sequences/heldout-set-sequences.csv"
MANIFEST_FILE="/a0/bitcore/workspace/data/MANIFEST.yaml"

# Validate GDPa1 file
echo "Validating GDPa1_v1.2_sequences.csv..."
if [ -f "$GDPa1_FILE" ]; then
    CURRENT_HASH=$(sha256sum "$GDPa1_FILE" | cut -d' ' -f1)
    if [ "$CURRENT_HASH" = "$GDPa1_HASH" ]; then
        echo "✓ VALID (Hash: $CURRENT_HASH)"
    else
        echo "✗ INVALID (Current: $CURRENT_HASH, Expected: $GDPa1_HASH)"
    fi
else
    echo "✗ FILE NOT FOUND"
fi
echo ""

# Validate heldout file
echo "Validating heldout-set-sequences.csv..."
if [ -f "$HELDOUT_FILE" ]; then
    CURRENT_HASH=$(sha256sum "$HELDOUT_FILE" | cut -d' ' -f1)
    if [ "$CURRENT_HASH" = "$HELDOUT_HASH" ]; then
        echo "✓ VALID (Hash: $CURRENT_HASH)"
    else
        echo "✗ INVALID (Current: $CURRENT_HASH, Expected: $HELDOUT_HASH)"
    fi
else
    echo "✗ FILE NOT FOUND"
fi
echo ""

# Check for MANIFEST.yaml
echo "Checking for MANIFEST.yaml..."
if [ -f "$MANIFEST_FILE" ]; then
    echo "✓ MANIFEST.yaml found"
    MANIFEST_HASH=$(sha256sum "$MANIFEST_FILE" | cut -d' ' -f1)
    echo "  Hash: $MANIFEST_HASH"
else
    echo "✗ MANIFEST.yaml not found"
fi
echo ""

# List data directories for reference
echo "=== Data Directory Structure ==="
find /a0/bitcore/workspace/data -type d | head -10
echo ""

echo "=== Validation Complete ==="
