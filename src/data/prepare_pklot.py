# =========================================================
# File name: prepare_pklot.py
# Project: Smart Parking Occupancy Detection
# Description:
#   Convert PKLot COCO annotations into cropped parking-slot images
#   for CNN-based parking occupancy classification.
# =========================================================

import json
import shutil
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile, ImageOps
from tqdm import tqdm


# =========================================================
# Global configuration
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PKLOT_DIR = PROJECT_ROOT / "data" / "raw" / "pklot"

# Use "sample" only for quick testing.
# Use "full" for real PKLot 5-fold cross-validation.
RUN_MODE = "full"

if RUN_MODE == "sample":
    PROCESSED_PKLOT_DIR = PROJECT_ROOT / "data" / "processed" / "pklot_sample"
    MAX_IMAGES_PER_SPLIT = 100
else:
    PROCESSED_PKLOT_DIR = PROJECT_ROOT / "data" / "processed" / "pklot"
    MAX_IMAGES_PER_SPLIT = None

CROP_DIR = PROCESSED_PKLOT_DIR / "crops"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "pklot"

SPLITS = ["train", "valid", "test"]

# COCO category names from your PKLot dataset
CATEGORY_TO_LABEL = {
    "space-empty": "free",
    "space-occupied": "occupied"
}

# Ignore very small or invalid bounding boxes
MIN_BOX_WIDTH = 8
MIN_BOX_HEIGHT = 8

# Save cropped slots as 150x150 to match CNRPark patch size and reduce disk size
SAVE_RESIZED_CROPS = True
CROP_SIZE = (150, 150)

JPEG_QUALITY = 90

# Keep False for safety. Set True only when you want to remove old PKLot processed output.
RESET_OUTPUT_FOLDER = False

ImageFile.LOAD_TRUNCATED_IMAGES = True


OUTPUT_COLUMNS = [
    "image_path",
    "label",
    "source_dataset",
    "source_split",
    "original_image",
    "image_id",
    "annotation_id",
    "category_id",
    "category_name",
    "original_width",
    "original_height",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2"
]


# =========================================================
# Folder and file functions
# =========================================================
def resetOutputFolder():
    if RESET_OUTPUT_FOLDER and PROCESSED_PKLOT_DIR.exists():
        shutil.rmtree(PROCESSED_PKLOT_DIR)


def createOutputFolders():
    PROCESSED_PKLOT_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in SPLITS:
        for label_name in ["free", "occupied"]:
            (CROP_DIR / split_name / label_name).mkdir(parents=True, exist_ok=True)


def checkRawDataset():
    if not RAW_PKLOT_DIR.exists():
        raise FileNotFoundError(f"PKLot raw folder not found: {RAW_PKLOT_DIR}")

    for split_name in SPLITS:
        split_dir = RAW_PKLOT_DIR / split_name
        annotation_path = split_dir / "_annotations.coco.json"

        if not split_dir.exists():
            raise FileNotFoundError(f"PKLot split folder not found: {split_dir}")

        if not annotation_path.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")

    print("PKLot raw dataset structure is valid.")


def loadCocoJson(split_name):
    annotation_path = RAW_PKLOT_DIR / split_name / "_annotations.coco.json"

    with open(annotation_path, "r", encoding="utf-8") as file:
        coco_data = json.load(file)

    return coco_data


def resolveImagePath(split_dir, file_name):
    image_path = split_dir / file_name

    if image_path.exists():
        return image_path

    image_path_by_name = split_dir / Path(file_name).name

    if image_path_by_name.exists():
        return image_path_by_name

    matches = list(split_dir.rglob(Path(file_name).name))

    if len(matches) > 0:
        return matches[0]

    return None


# =========================================================
# COCO processing functions
# =========================================================
def getCategoryIdToName(coco_data):
    category_id_to_name = {}

    for category in coco_data.get("categories", []):
        category_id = category["id"]
        category_name = category["name"]
        category_id_to_name[category_id] = category_name

    return category_id_to_name


def groupUsefulAnnotationsByImage(coco_data, category_id_to_name):
    annotations_by_image = {}
    useful_annotation_count = 0
    skipped_category_count = 0

    for annotation in coco_data.get("annotations", []):
        category_id = annotation.get("category_id")
        category_name = category_id_to_name.get(category_id)

        if category_name not in CATEGORY_TO_LABEL:
            skipped_category_count += 1
            continue

        image_id = annotation.get("image_id")

        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []

        annotations_by_image[image_id].append(annotation)
        useful_annotation_count += 1

    return annotations_by_image, useful_annotation_count, skipped_category_count


def limitImagesIfNeeded(images):
    if MAX_IMAGES_PER_SPLIT is None:
        return images

    return images[:MAX_IMAGES_PER_SPLIT]


def clipBoundingBox(bbox, image_width, image_height):
    if bbox is None or len(bbox) < 4:
        return None

    x, y, width, height = bbox

    if width <= 0 or height <= 0:
        return None

    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + width))
    y2 = int(round(y + height))

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    if x2 <= x1 or y2 <= y1:
        return None

    crop_width = x2 - x1
    crop_height = y2 - y1

    if crop_width < MIN_BOX_WIDTH or crop_height < MIN_BOX_HEIGHT:
        return None

    return x1, y1, x2, y2


def saveCrop(cropped_image, split_name, label_name, image_id, annotation_id):
    crop_filename = f"{split_name}_img{image_id}_ann{annotation_id}.jpg"
    crop_path = CROP_DIR / split_name / label_name / crop_filename

    if SAVE_RESIZED_CROPS:
        cropped_image = cropped_image.resize(CROP_SIZE, Image.Resampling.BILINEAR)

    cropped_image.save(
        crop_path,
        format="JPEG",
        quality=JPEG_QUALITY,
        optimize=True
    )

    return crop_path


# =========================================================
# Main processing functions
# =========================================================
def processOneSplit(split_name):
    print("\n" + "=" * 80)
    print(f"Processing PKLot split: {split_name}")
    print("=" * 80)

    split_dir = RAW_PKLOT_DIR / split_name

    coco_data = loadCocoJson(split_name)
    category_id_to_name = getCategoryIdToName(coco_data)

    annotations_by_image, useful_annotation_count, skipped_category_count = (
        groupUsefulAnnotationsByImage(
            coco_data=coco_data,
            category_id_to_name=category_id_to_name
        )
    )

    images = coco_data.get("images", [])
    images = limitImagesIfNeeded(images)

    print(f"Images to process        : {len(images)}")
    print(f"Total annotations in JSON: {len(coco_data.get('annotations', []))}")
    print(f"Useful annotations       : {useful_annotation_count}")
    print(f"Skipped category annots  : {skipped_category_count}")

    rows = []

    missing_images = 0
    unreadable_images = 0
    skipped_boxes = 0
    processed_images = 0

    for image_info in tqdm(images, desc=f"Cropping {split_name}"):
        image_id = image_info.get("id")
        file_name = image_info.get("file_name")

        image_path = resolveImagePath(
            split_dir=split_dir,
            file_name=file_name
        )

        if image_path is None:
            missing_images += 1
            continue

        image_annotations = annotations_by_image.get(image_id, [])

        if len(image_annotations) == 0:
            continue

        try:
            with Image.open(image_path) as opened_image:
                image = ImageOps.exif_transpose(opened_image).convert("RGB")
        except Exception:
            unreadable_images += 1
            continue

        image_width, image_height = image.size
        processed_images += 1

        for annotation in image_annotations:
            category_id = annotation.get("category_id")
            category_name = category_id_to_name.get(category_id)
            label_name = CATEGORY_TO_LABEL[category_name]

            bbox = annotation.get("bbox")
            clipped_box = clipBoundingBox(
                bbox=bbox,
                image_width=image_width,
                image_height=image_height
            )

            if clipped_box is None:
                skipped_boxes += 1
                continue

            x1, y1, x2, y2 = clipped_box

            cropped_image = image.crop((x1, y1, x2, y2))

            annotation_id = annotation.get("id")

            crop_path = saveCrop(
                cropped_image=cropped_image,
                split_name=split_name,
                label_name=label_name,
                image_id=image_id,
                annotation_id=annotation_id
            )

            rows.append({
                "image_path": str(crop_path),
                "label": label_name,
                "source_dataset": "PKLot",
                "source_split": split_name,
                "original_image": str(image_path),
                "image_id": image_id,
                "annotation_id": annotation_id,
                "category_id": category_id,
                "category_name": category_name,
                "original_width": image_width,
                "original_height": image_height,
                "bbox_x": bbox[0],
                "bbox_y": bbox[1],
                "bbox_width": bbox[2],
                "bbox_height": bbox[3],
                "crop_x1": x1,
                "crop_y1": y1,
                "crop_x2": x2,
                "crop_y2": y2
            })

    split_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    split_csv_path = PROCESSED_PKLOT_DIR / f"{split_name}.csv"
    split_df.to_csv(split_csv_path, index=False)

    label_counts = split_df["label"].value_counts().to_dict()

    split_stats = {
        "split": split_name,
        "images_in_json": len(coco_data.get("images", [])),
        "images_processed_limit": len(images),
        "images_opened": processed_images,
        "missing_images": missing_images,
        "unreadable_images": unreadable_images,
        "useful_annotations": useful_annotation_count,
        "skipped_category_annotations": skipped_category_count,
        "skipped_invalid_boxes": skipped_boxes,
        "created_crops": len(split_df),
        "free_count": label_counts.get("free", 0),
        "occupied_count": label_counts.get("occupied", 0),
        "csv_path": str(split_csv_path)
    }

    print(f"\nFinished split: {split_name}")
    print(f"Saved CSV     : {split_csv_path}")
    print(f"Created crops : {len(split_df)}")
    print(f"Free          : {label_counts.get('free', 0)}")
    print(f"Occupied      : {label_counts.get('occupied', 0)}")
    print(f"Missing images: {missing_images}")
    print(f"Unreadable    : {unreadable_images}")
    print(f"Skipped boxes : {skipped_boxes}")

    return split_df, split_stats


def preparePKLotDataset():
    start_time = time.time()

    resetOutputFolder()
    createOutputFolders()
    checkRawDataset()

    print("=" * 80)
    print("Preparing PKLot Dataset")
    print("=" * 80)
    print(f"Run mode           : {RUN_MODE}")
    print(f"Raw PKLot directory: {RAW_PKLOT_DIR}")
    print(f"Processed directory: {PROCESSED_PKLOT_DIR}")
    print(f"Crop directory     : {CROP_DIR}")
    print(f"Max images/split   : {MAX_IMAGES_PER_SPLIT}")
    print(f"Save resized crops : {SAVE_RESIZED_CROPS}")
    print(f"Crop size          : {CROP_SIZE}")
    print("=" * 80)

    all_dataframes = []
    all_stats = []

    for split_name in SPLITS:
        split_df, split_stats = processOneSplit(split_name)
        all_dataframes.append(split_df)
        all_stats.append(split_stats)

    all_df = pd.concat(all_dataframes, ignore_index=True)

    all_csv_path = PROCESSED_PKLOT_DIR / "pklot_all.csv"
    all_df.to_csv(all_csv_path, index=False)

    label_summary_path = REPORT_DIR / f"pklot_prepare_summary_{RUN_MODE}.csv"
    stats_path = REPORT_DIR / f"pklot_prepare_stats_{RUN_MODE}.csv"

    if len(all_df) > 0:
        label_summary_df = (
            all_df
            .groupby(["source_split", "label"])
            .size()
            .reset_index(name="count")
        )
    else:
        label_summary_df = pd.DataFrame(
            columns=["source_split", "label", "count"]
        )

    label_summary_df.to_csv(label_summary_path, index=False)

    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(stats_path, index=False)

    total_time = (time.time() - start_time) / 60

    print("\n" + "=" * 80)
    print("PKLot Preparation Completed")
    print("=" * 80)
    print(f"Total cropped samples: {len(all_df)}")
    print(f"Saved all CSV        : {all_csv_path}")
    print(f"Saved label summary  : {label_summary_path}")
    print(f"Saved process stats  : {stats_path}")
    print(f"Total time           : {total_time:.2f} minutes")

    if len(all_df) > 0:
        print("\nOverall label distribution:")
        print(all_df["label"].value_counts())


# =========================================================
# Main function
# =========================================================
def main():
    preparePKLotDataset()


if __name__ == "__main__":
    main()
