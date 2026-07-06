from pathlib import Path
import pandas as pd

# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(r"D:\smart-parking-occupancy-detection")

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "cnrpark_ext"
CSV_PATH = DATASET_DIR / "CNRPark+EXT.csv"
IMAGE_ROOT = DATASET_DIR / "CNR-EXT-Patches-150x150"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


# =========================================================
# 2. LOAD CSV
# =========================================================

def load_csv():
    if not CSV_PATH.exists():
        print(f"[ERROR] CSV file not found: {CSV_PATH}")
        return None

    try:
        df = pd.read_csv(CSV_PATH, sep=None, engine="python")
        return df
    except Exception as error:
        print(f"[ERROR] Cannot read CSV: {error}")
        return None


# =========================================================
# 3. FIND IMAGES
# =========================================================

def find_images():
    if not IMAGE_ROOT.exists():
        print(f"[ERROR] Image folder not found: {IMAGE_ROOT}")
        return []

    image_paths = []

    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(IMAGE_ROOT.rglob(f"*{ext}"))

    return image_paths


# =========================================================
# 4. MAIN FUNCTION
# =========================================================

def main():
    print("=" * 80)
    print("CNRPark+EXT Dataset Checking")
    print("=" * 80)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset dir  : {DATASET_DIR}")
    print(f"CSV path     : {CSV_PATH}")
    print(f"Image root   : {IMAGE_ROOT}")

    print("\n" + "=" * 80)
    print("1. CSV CHECK")
    print("=" * 80)

    df = load_csv()

    if df is not None:
        print(f"CSV loaded successfully.")
        print(f"Rows   : {len(df)}")
        print(f"Columns: {len(df.columns)}")

        print("\nColumn names:")
        for col in df.columns:
            print(f"- {col}")

        print("\nFirst 5 rows:")
        print(df.head())
    else:
        print("CSV loading failed.")

    print("\n" + "=" * 80)
    print("2. IMAGE CHECK")
    print("=" * 80)

    image_paths = find_images()

    print(f"Total images found: {len(image_paths)}")

    print("\nFirst 10 image paths:")
    for path in image_paths[:10]:
        print(path)

    print("\nTop-level folders inside image root:")

    if IMAGE_ROOT.exists():
        for item in IMAGE_ROOT.iterdir():
            if item.is_dir():
                print(f"- {item.name}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if df is not None and len(image_paths) > 0:
        print("Dataset check successful.")
        print("Next step: prepare train/validation/test split.")
    else:
        print("Dataset check failed. Please check paths or extraction.")

    print("\nDone.")


if __name__ == "__main__":
    main()
