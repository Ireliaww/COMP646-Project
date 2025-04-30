import requests, json, os, math, sys, random

# --------------------------
# Configuration
# --------------------------
# Total number of images desired
TOTAL_IMAGES = 1000
# List of WikiArt styles (genre identifiers)
styles = ["impressionismus",
          "realismus",
          "romantik",
          "expressionismus",
          "post-impressionismus",
          "surrealismus",
          "art-nouveau-modern",
          "barock",
          "symbolismus",
          "abstrakter-expressionismus",
          "naive-kunst-primitivismus",
          "neoklassizistismus",
          "rokkoko",
          "nordliche-renaissance",
          "kubismus",
          "minimalismus",
          "pop-art",
          "informel",
          "abstrakte-kunst",
          "farbfeldmalerei",
          "ukiyo-e",
          "manierismus-spatrenaissance",
          "hochrenaissance",
          "fruhrenaissance",
          "konzeptuelle-kunst",
          "magischer-realismus",
          "neoexpressionismus",
          "op-art",
          "lyrische-abstraktion",
          "akademische-kunst",
          "zeitgenossischer-realismus",
          "art-deco",
          "fauvismus",
          "konkretismus",
          "sumi-e",
          "post-minimalismus",
          "hard-edge-harte-kante",
          "neoromantik",
          "tachismus",
          "sozialer-realismus",
          "pointilismus",
          "sosaku-hanga",
          "naturalismus",
          "neodada",
          "konstruktivismus",
          "dada",
          "orientalismus",
          "shin-hanga",
          "luminismus",
          "neuer-realismus",
          "regionalismus",
          "futurismus",
          "divisionismus",
          "fantastischer-realismus",
          "prazisionismus",
          "nachmalerische-abstraktion",
          "protorenaissance",
          "art-brut",
          "nouveau-realisme",
          "sozialistischer-realismus",
          "feministische-kunst",
          "neo-pop-art",
          "amerikanischer-realismus",
          "streetart",
          "orphismus",
          "licht-und-raum",
          "tonalismus",
          "neominimalismus",
          "photorealismus",
          "kinetische-kunst",
          "klassizismus",
          "internationale-gothik",
          "tenebrismus",
          "metaphysische-kunst",
          "pictorialismus",
          "synthetischer-kubismus",
          "cloisonismus",
          "japanismus",
          "neue-europaische-kunst",
          "neoplastizismus",
          "kitsch",
          "kubo-futurismus",
          "pusimus",
          "zen",
          "muralismo",
          "raumkunst",
          "neobarock",
          "p-d-pattern-and-decoration-muster-und-dekoration",
          "neo-geo",
          "suprematismus",
          "biedermeier",
          "byzantinisch",
          "umweltkunst",
          "analytischer-kubismus",
          "intimismus",
          "art-brut-rohe-kunst",
          "action-painting",
          "neo-rokkoko",
          "romanisch",
          "neokonkretismus",
          "analytischer-realismus",
          "verismo",
          "mozarabisch",
          "transautomatismus",
          "modernismo-0",
          "hyperrealismus",
          "ottomanische-zeit",
          "mechanistischer-kubismus",
          "safavid-zeit",
          "lowbrow-kunst",
          "figurativer-experessionismus",
          "maximalismus",
          "neo-suprematism",
          "nanga-bunjinga",
          "maaslandische-kunst",
          "neofigurative-kunst",
          "letterismus",
          "automatische-malerei",
          "synthetismus",
          "kartographische-kunst",
          "neuer-kausalismus",
          "trashart",
          "posterkunst-realismus",
          "indigene-kunst",
          "nihonga",
          "gongbi",
          "primitivismus",
          "existentialistische-kunst",
          "timurid-zeit",
          "stuckismus",
          "cyber-art",
          "indische-weltraummalerei",
          "tubismus",
          "dialektische-kunst",
          "superflat",
          "kostumbrismus",
          "neobyzantinisch",
          "transavantgarde",
          "hypermanierismus-anachronismus",
          "gewebekunst",
          "mail-art-postkunst",
          "mogulreich",
          "ilkhanid",
          "toyisme",
          "nastaliq",
          "joseon-zeit",
          "yamato-e",
          "synchronismus",
          "gothisch",
          "art-singulier",
          "kubo-expressionismus",
          "strassenphotographie",
          "dustere-kunst",
          "perzeptismus",
          "rayonismus",
          "spektralismus",
          "renaissance",
          "sky-art"]
# Compute per-style quota
num_styles = len(styles)
per_style = math.ceil(TOTAL_IMAGES / num_styles)

# Root folder where images will be saved
dst_root = os.path.expanduser("~/wikiartData/")
os.makedirs(dst_root, exist_ok=True)

# Progress helper
def print_progress(msg):
    sys.stdout.write(f"\r{msg}")
    sys.stdout.flush()

# --------------------------
# Functions to fetch metadata
# --------------------------

def getData(style, page):
    """
    Request JSON metadata for a particular style and page.
    """
    url = f'https://www.wikiart.org/en/paintings-by-style/{style}'
    payload = {'json': 2, 'page': page}
    res = requests.post(url, payload)
    while res.status_code != 200:
        res = requests.post(url, payload)
    return res.json()


def getAllPaintings(style):
    """
    Retrieve all painting metadata for a given style.
    Returns a list of metadata dicts.
    """
    init = getData(style, 1)
    total_count = init['AllPaintingsCount']
    page_size = init['PageSize']
    pages = math.ceil(total_count / page_size)
    paintings = init['Paintings']
    for page in range(2, pages + 1):
        page_data = getData(style, page)['Paintings']
        # retry if data seems incomplete
        while len(page_data) < page_size and page < pages:
            page_data = getData(style, page)['Paintings']
        paintings.extend(page_data)
    return paintings

# --------------------------
# Sampling and Download
# --------------------------

download_count = 0
for style in styles:
    print(f"Processing style '{style}' ({download_count}/{TOTAL_IMAGES} downloaded)")
    # Fetch metadata once per style
    all_meta = getAllPaintings(style)
    # Randomly sample up to per_style items
    random.shuffle(all_meta)
    subset = all_meta[:per_style]

    # Create folder for this style
    style_dir = os.path.join(dst_root, style)
    os.makedirs(style_dir, exist_ok=True)

    # Download images
    for meta in subset:
        if download_count >= TOTAL_IMAGES:
            break
        img_url = meta.get('image')
        img_name = os.path.basename(img_url)
        dst_path = os.path.join(style_dir, img_name)
        if not os.path.isfile(dst_path):
            response = requests.get(img_url)
            while response.status_code != 200:
                response = requests.get(img_url)
            with open(dst_path, 'wb') as f:
                f.write(response.content)
            download_count += 1
            print_progress(f"Downloaded {download_count}/{TOTAL_IMAGES}")

    if download_count >= TOTAL_IMAGES:
        break

print("\nDone! Total images downloaded:", download_count)
