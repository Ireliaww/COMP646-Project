# --------------------------
# data/download_wikiart.py
# --------------------------
"""
Script to download WikiArt dataset using Kaggle API.
Requires: pip install kaggle, and set KAGGLE_USERNAME & KAGGLE_KEY as env vars.
Replace `dataset_slug` with the actual Kaggle dataset identifier.
"""
import os
import kagglehub

path = kagglehub.dataset_download("steubk/wikiart")
import zipfile

api = KaggleApi()
api.authenticate()

dataset_slug = 'bryanlim75/wikiart'  # TODO: Replace with actual slug
output_path = 'data/wikiart'
os.makedirs(output_path, exist_ok=True)
api.dataset_download_files(dataset_slug, path=output_path, unzip=False)
zip_path = os.path.join(output_path, f"{dataset_slug.split('/')[-1]}.zip")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_path)
print("Download and extraction complete.")