# =========================================================
# File: crop_cnn_external.py
# Crop Free/Occupied spaces from the original Getty image
# =========================================================


# =========================================================
# 1. LIBRARIES
# =========================================================

from pathlib import Path

import cv2


# =========================================================
# 2. PATHS AND SETTINGS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "demo"
    / "input"
    / "web"
    / "web_01.jpg"
)

SOURCE_NAME = "web"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "demo"
    / "input"
    / "cnn_external_report"
)

FREE_FOLDER = OUTPUT_ROOT / "free"
OCCUPIED_FOLDER = OUTPUT_ROOT / "occupied"

MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 800


# =========================================================
# 3. SUPPORTING FUNCTIONS
# =========================================================

def get_next_output_path(
    folder: Path,
    source_name: str,
    label: str,
) -> Path:
    """
    Generate filenames such as:
    web_free_001.jpg
    web_occupied_001.jpg
    """

    index = 1

    while True:
        output_path = (
            folder
            / f"{source_name}_{label}_{index:03d}.jpg"
        )

        if not output_path.exists():
            return output_path

        index += 1


def resize_for_display(image):
    """
    Resize the image only for display.
    Crops are taken from the original image.
    """

    original_height, original_width = (
        image.shape[:2]
    )

    display_scale = min(
        MAX_DISPLAY_WIDTH / original_width,
        MAX_DISPLAY_HEIGHT / original_height,
        1.0,
    )

    display_width = int(
        original_width * display_scale
    )

    display_height = int(
        original_height * display_scale
    )

    display_image = cv2.resize(
        image,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA,
    )

    return display_image, display_scale


def crop_parking_spaces() -> None:
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Original image not found:\n{IMAGE_PATH}"
        )

    FREE_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    OCCUPIED_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_image = cv2.imread(
        str(IMAGE_PATH)
    )

    if original_image is None:
        raise RuntimeError(
            f"Cannot read image:\n{IMAGE_PATH}"
        )

    preview_image, display_scale = (
        resize_for_display(original_image)
    )

    print("=" * 65)
    print("EXTERNAL CNN CROP TOOL")
    print("=" * 65)
    print(f"Source : {SOURCE_NAME}")
    print(f"Image  : {IMAGE_PATH}")
    print(f"Output : {OUTPUT_ROOT}")
    print()
    print("Drag around ONE parking space.")
    print("Press ENTER or SPACE to confirm.")
    print("Enter F for Free or O for Occupied.")
    print("Press C or ESC in the image window to finish.")
    print("=" * 65)

    while True:
        window_name = (
            "Select one parking space | "
            "ENTER: confirm | C/ESC: finish"
        )

        x, y, width, height = cv2.selectROI(
            window_name,
            preview_image,
            showCrosshair=True,
            fromCenter=False,
        )

        cv2.destroyWindow(window_name)

        if width == 0 or height == 0:
            print("\nCropping completed.")
            break

        label_input = input(
            "Label [F = Free, O = Occupied, "
            "S = Skip, Q = Quit]: "
        ).strip().lower()

        if label_input == "q":
            print("\nCropping completed.")
            break

        if label_input == "s":
            print("Crop skipped.\n")
            continue

        if label_input not in {"f", "o"}:
            print("Invalid label. Crop skipped.\n")
            continue

        original_x = int(
            x / display_scale
        )

        original_y = int(
            y / display_scale
        )

        original_width = int(
            width / display_scale
        )

        original_height = int(
            height / display_scale
        )

        x_end = min(
            original_x + original_width,
            original_image.shape[1],
        )

        y_end = min(
            original_y + original_height,
            original_image.shape[0],
        )

        crop = original_image[
            original_y:y_end,
            original_x:x_end,
        ]

        if crop.size == 0:
            print("Invalid crop. Try again.\n")
            continue

        if label_input == "f":
            label = "free"
            output_folder = FREE_FOLDER
            rectangle_color = (0, 255, 0)

        else:
            label = "occupied"
            output_folder = OCCUPIED_FOLDER
            rectangle_color = (0, 0, 255)

        output_path = get_next_output_path(
            folder=output_folder,
            source_name=SOURCE_NAME,
            label=label,
        )

        saved = cv2.imwrite(
            str(output_path),
            crop,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        if not saved:
            raise RuntimeError(
                f"Failed to save crop:\n{output_path}"
            )

        cv2.rectangle(
            preview_image,
            (x, y),
            (x + width, y + height),
            rectangle_color,
            2,
        )

        print(f"Saved: {output_path.name}")
        print(
            f"Crop size: "
            f"{crop.shape[1]} x {crop.shape[0]}\n"
        )

    cv2.destroyAllWindows()


# =========================================================
# 4. MAIN
# =========================================================

if __name__ == "__main__":
    crop_parking_spaces()