import requests, json, os, math, sys, random, time, csv

# --------------------------
# Configuration
# --------------------------
TOTAL_IMAGES = 2000
styles = [
    "gothisch",
    "protorenaissance",
    "fruhrenaissance",
    "hochrenaissance",
    "manierismus-spatrenaissance",
    "barock",
    "rokkoko",
    "neoklassizistismus",
    "romantik",
    "realismus",
    "impressionismus",
    "post-impressionismus",
    "expressionismus",
    "abstrakter-expressionismus",
]

num_styles = len(styles)
per_style = math.ceil(TOTAL_IMAGES / num_styles)

dst_root = os.path.expanduser("../Datasets/Wiki")
os.makedirs(dst_root, exist_ok=True)

# Initialize list to record downloaded samples
downloaded_list = []

# Helper: print progress in-place
def print_progress(msg):
    sys.stdout.write(f"\r{msg}")
    sys.stdout.flush()

# --------------------------
# Robust data fetch
# --------------------------
def getData(style, page, max_retries=5):
    url = f'https://www.wikiart.org/en/paintings-by-style/{style}'
    payload = {'json': 2, 'page': page}
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.post(url, data=payload, timeout=10)
            if res.status_code != 200:
                time.sleep(1)
                continue
            return res.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"\nWarning: error on style={style}, page={page}, attempt {attempt}: {e}")
            time.sleep(1)
    raise RuntimeError(f"Failed to fetch metadata for style={style}, page={page}")

# --------------------------
# Collect all metadata for a style
# --------------------------
def getAllPaintings(style):
    init = getData(style, 1)
    total_count = init.get('AllPaintingsCount', 0)
    page_size = init.get('PageSize', len(init.get('Paintings', [])))
    pages = math.ceil(total_count / page_size) if page_size else 0
    paintings = init.get('Paintings', [])
    for page in range(2, pages + 1):
        print_progress(f"Fetching {style} page {page}/{pages}")
        page_data = getData(style, page).get('Paintings', [])
        paintings.extend(page_data)
    return paintings

# --------------------------
# Sample & Download Images
# --------------------------
download_count = 0
for style in styles:
    if download_count >= TOTAL_IMAGES:
        break
    print(f"\nProcessing style '{style}' ({download_count}/{TOTAL_IMAGES})")
    try:
        metadata_list = getAllPaintings(style)
    except Exception as e:
        print(f"Error fetching style {style}: {e}, skipping.")
        continue

    random.shuffle(metadata_list)
    subset = metadata_list[:per_style]

    style_dir = os.path.join(dst_root, style)
    os.makedirs(style_dir, exist_ok=True)

    for meta in subset:
        if download_count >= TOTAL_IMAGES:
            break
        img_url = meta.get('image')
        if not img_url:
            continue
        img_name = os.path.basename(img_url)
        dst_path = os.path.join(style_dir, img_name)
        # Record whether new or existing
        if os.path.exists(dst_path):
            downloaded_list.append((style, img_name))
            download_count += 1
            continue
        try:
            resp = requests.get(img_url, timeout=10)
            if resp.status_code == 200:
                with open(dst_path, 'wb') as f:
                    f.write(resp.content)
                downloaded_list.append((style, img_name))
                download_count += 1
                print_progress(f"Downloaded {download_count}/{TOTAL_IMAGES}")
        except requests.exceptions.RequestException as e:
            print(f"\nWarning: failed to download {img_url}: {e}")

print(f"\nDownload complete. Total images: {download_count}")

# --------------------------
# Write annotations CSV
# --------------------------
csv_path = os.path.join(dst_root, 'annotations.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['style', 'image_filename'])
    for style, img_name in downloaded_list:
        writer.writerow([style, img_name])
print(f"Annotations CSV saved to {csv_path}")
