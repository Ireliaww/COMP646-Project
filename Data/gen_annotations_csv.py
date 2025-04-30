# File: scripts/gen_annotations_csv.py

import os
import csv

# Root of your Datasets directory (must match your real folder name & casing)
DATASETS_ROOT = '../Datasets/Wiki'
# Ensure the directory exists
os.makedirs(DATASETS_ROOT, exist_ok=True)

# Output CSV path
CSV_PATH = os.path.join(DATASETS_ROOT, 'annotations.csv')

with open(CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(['style', 'image_filename'])

    # Iterate over each style subfolder
    for style in os.listdir(DATASETS_ROOT):
        style_dir = os.path.join(DATASETS_ROOT, style)
        if not os.path.isdir(style_dir):
            continue

        # Iterate over all files in that style folder
        for fname in os.listdir(style_dir):
            # Skip non-image files (adjust extensions as needed)
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            # Write one row: style + image filename
            writer.writerow([style, fname])

print(f"Annotations CSV generated at: {CSV_PATH}")
