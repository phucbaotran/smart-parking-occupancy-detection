# ============================================================
# setup_roi_folder.py
#
# Select parking-slot ROIs for every image in a folder.
#
# Each image receives:
#   demo/iu/config/<image_name>_slots.json
#   demo/iu/config/<image_name>_slots_preview.jpg
#
# The script reuses setup_rois() from demo_parking_roi_v3.py.
# ============================================================

import argparse
from pathlib import Path

from demo_parking_roi_v3 import (
    SUPPORTED_IMAGE_SUFFIXES,
    setup_rois,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "demo"
    / "iu"
    / "input"
)

DEFAULT_CONFIG_DIR = (
    PROJECT_ROOT
    / "demo"
    / "iu"
    / "config"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw parking ROIs for every IU image in a folder."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing IU images.",
    )

    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Folder used to save ROI JSON and preview images.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Redraw ROIs even when a JSON config already exists.",
    )

    return parser.parse_args()


def collect_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input folder was not found:\n{input_dir}"
        )

    images = sorted(
        path.resolve()
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )

    if not images:
        raise FileNotFoundError(
            f"No supported images were found in:\n{input_dir}"
        )

    return images


def main() -> None:
    args = parse_arguments()

    input_dir = args.input_dir.resolve()
    config_dir = args.config_dir.resolve()

    config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = collect_images(input_dir)

    print("=" * 78)
    print("IU FOLDER ROI SETUP")
    print("=" * 78)
    print(f"Input folder       : {input_dir}")
    print(f"Config folder      : {config_dir}")
    print(f"Images found       : {len(images)}")
    print("=" * 78)

    completed = 0
    skipped = 0
    failed = 0

    for index, image_path in enumerate(images, start=1):
        config_path = (
            config_dir
            / f"{image_path.stem}_slots.json"
        )

        print()
        print("-" * 78)
        print(
            f"[{index}/{len(images)}] "
            f"{image_path.name}"
        )
        print("-" * 78)

        if config_path.exists() and not args.force:
            print(
                "SKIPPED: ROI config already exists.\n"
                f"{config_path}"
            )
            skipped += 1
            continue

        try:
            setup_rois(
                source=image_path,
                config_path=config_path,
            )
            completed += 1

        except KeyboardInterrupt:
            print("\nROI setup stopped by user.")
            break

        except Exception as exc:
            print(
                f"FAILED: {type(exc).__name__}: {exc}"
            )
            failed += 1

    print()
    print("=" * 78)
    print("FOLDER ROI SETUP FINISHED")
    print("=" * 78)
    print(f"Completed          : {completed}")
    print(f"Skipped            : {skipped}")
    print(f"Failed             : {failed}")
    print(f"Config folder      : {config_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
