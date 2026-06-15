#!/bin/bash
# Download How2Sign B-F-H 2D Keypoints + English text annotations.
# Usage:
#   bash scripts/download_how2sign_keypoints.sh /path/to/target/dir
#
# This downloads ~23 GB (train+val+test) of frontal-view keypoints
# plus the re-aligned English sentence CSVs.

set -euo pipefail

TARGET="${1:-./data/how2sign}"
mkdir -p "$TARGET"

echo "Downloading How2Sign B-F-H 2D keypoints + text to $TARGET"

# ---- Keypoints (frontal view) ----
# Train
echo "Downloading train keypoints (21 GB) ..."
gdown --fuzzy "https://drive.google.com/file/d/1lnsDN-LxcsroOmetdG5_sXYXZ7setlS4/view" -O "$TARGET/train_2D_keypoints.tar.gz"

# Val
echo "Downloading val keypoints (1.2 GB) ..."
gdown --fuzzy "https://drive.google.com/file/d/1aOhRknNWj8APdxHmwJdQrMo5xuIGNXxM/view" -O "$TARGET/val_2D_keypoints.tar.gz"

# Test
echo "Downloading test keypoints (1.6 GB) ..."
gdown --fuzzy "https://drive.google.com/file/d/1quj8Ipm56pH65KAKK3Pc-sqZ0ozw2gSe/view" -O "$TARGET/test_2D_keypoints.tar.gz"

# ---- English text CSVs (re-aligned) ----
echo "Downloading text CSVs ..."
gdown --fuzzy "https://drive.google.com/file/d/1BWt2ASmOIUM8tWnCuRtl9AdQsp4zsjPN/view" -O "$TARGET/how2sign_realigned_train.csv"
gdown --fuzzy "https://drive.google.com/file/d/1hcPqXfHIHHGHUQYfT3eFARTIgi0M1C9_/view" -O "$TARGET/how2sign_realigned_val.csv"
gdown --fuzzy "https://drive.google.com/file/d/1OsTvMsVFOMk54r65v2gcxOottV41ZEUo/view" -O "$TARGET/how2sign_realigned_test.csv"

echo "Extracting keypoints ..."
tar -xf "$TARGET/train_2D_keypoints.tar.gz" -C "$TARGET" && rm -f "$TARGET/train_2D_keypoints.tar.gz"
tar -xf "$TARGET/val_2D_keypoints.tar.gz"   -C "$TARGET" && rm -f "$TARGET/val_2D_keypoints.tar.gz"
tar -xf "$TARGET/test_2D_keypoints.tar.gz"  -C "$TARGET" && rm -f "$TARGET/test_2D_keypoints.tar.gz"

# Move CSVs into the expected directory structure
mkdir -p "$TARGET/sentence_level/train/text/en/raw_text/re_aligned"
mkdir -p "$TARGET/sentence_level/val/text/en/raw_text/re_aligned"
mkdir -p "$TARGET/sentence_level/test/text/en/raw_text/re_aligned"
mv "$TARGET/how2sign_realigned_train.csv" "$TARGET/sentence_level/train/text/en/raw_text/re_aligned/"
mv "$TARGET/how2sign_realigned_val.csv"   "$TARGET/sentence_level/val/text/en/raw_text/re_aligned/"
mv "$TARGET/how2sign_realigned_test.csv"  "$TARGET/sentence_level/test/text/en/raw_text/re_aligned/"

echo "Done! How2Sign data is ready at $TARGET"
echo "Folder structure:"
find "$TARGET/sentence_level" -maxdepth 4 -type d | head -20
