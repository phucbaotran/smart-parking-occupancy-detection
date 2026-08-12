# ============================================================
# demo_parking_roi.py
#
# Fast parking occupancy demo for:
#   - One campus image
#   - One fixed-camera parking video
#
# Method:
#   1. Use pretrained YOLO11 to detect vehicles.
#   2. Manually define parking-slot ROIs once.
#   3. A slot is occupied when a detected vehicle overlaps it.
#
# This script is intentionally separate from the custom PKLot
# free/occupied detector.
# ============================================================


# ============================================================
# 1. SUPPORTING LIBRARIES
# ============================================================

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHOW_VEHICLE_BOXES = True
SHOW_STATUS_LABELS = True
SHOW_INFERENCE_TIME = False


DEFAULT_CONFIG_DIR = PROJECT_ROOT / "demo" / "config"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "demo" / "output"

SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

SUPPORTED_VIDEO_SUFFIXES = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
    ".wmv",
}

VEHICLE_CLASS_NAMES = {
    "car",
    "motorcycle",
    "bus",
    "truck",
}


# ============================================================
# 3. ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Parking occupancy demo using predefined parking ROIs "
            "and pretrained YOLO11 vehicle detection."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to one image or one local video file.",
    )

    parser.add_argument(
        "--setup",
        action="store_true",
        help=(
            "Open the first frame and manually define parking-slot ROIs. "
            "The selected ROIs are saved to JSON."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "ROI JSON path. If omitted, the script uses "
            "demo/config/<source_name>_slots.json."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output image/video path. If omitted, it is saved "
            "inside demo/output."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained YOLO vehicle detector. Default: yolo11n.pt",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="YOLO inference image size.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Vehicle confidence threshold.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="YOLO Non-Maximum Suppression IoU threshold.",
    )

    parser.add_argument(
        "--slot-overlap",
        type=float,
        default=0.20,
        help=(
            "Minimum vehicle-box overlap ratio for marking a slot occupied. "
            "Default: 0.20"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Inference device: 0 for GPU or cpu for CPU.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the processed image/video while running.",
    )

    return parser.parse_args()


# ============================================================
# 4. SUPPORTING FUNCTIONS
# ============================================================

def validate_arguments(args: argparse.Namespace) -> None:
    """Validate file paths and numeric parameters."""

    args.source = args.source.resolve()

    if not args.source.is_file():
        raise FileNotFoundError(
            f"Source file was not found:\n{args.source}"
        )

    suffix = args.source.suffix.lower()

    if suffix not in SUPPORTED_IMAGE_SUFFIXES | SUPPORTED_VIDEO_SUFFIXES:
        raise ValueError(
            f"Unsupported source type: {suffix}"
        )

    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than 0.")

    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1.")

    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be between 0 and 1.")

    if not 0.0 <= args.slot_overlap <= 1.0:
        raise ValueError("--slot-overlap must be between 0 and 1.")

    DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.config is None:
        args.config = (
            DEFAULT_CONFIG_DIR
            / f"{args.source.stem}_slots.json"
        )
    else:
        args.config = args.config.resolve()

    if args.output is None:
        if suffix in SUPPORTED_IMAGE_SUFFIXES:
            args.output = (
                DEFAULT_OUTPUT_DIR
                / f"{args.source.stem}_roi_detected.jpg"
            )
        else:
            args.output = (
                DEFAULT_OUTPUT_DIR
                / f"{args.source.stem}_roi_detected.mp4"
            )
    else:
        args.output = args.output.resolve()

    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)


def source_is_image(source: Path) -> bool:
    """Return True when source is an image."""

    return source.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def read_first_frame(source: Path) -> np.ndarray:
    """Read an image or the first frame of a video."""

    if source_is_image(source):
        frame = cv2.imread(str(source))

        if frame is None:
            raise OSError(
                f"Could not read image:\n{source}"
            )

        return frame

    capture = cv2.VideoCapture(str(source))

    if not capture.isOpened():
        raise OSError(
            f"Could not open video:\n{source}"
        )

    ok, frame = capture.read()
    capture.release()

    if not ok or frame is None:
        raise OSError(
            f"Could not read the first video frame:\n{source}"
        )

    return frame


def resize_for_roi_selection(
    frame: np.ndarray,
    max_width: int = 1100,
    max_height: int = 650,
) -> tuple[np.ndarray, float]:
    """
    Resize only when the source is larger than the available ROI window.

    The image is never enlarged, so the ROI setup view remains sharp.
    """

    height, width = frame.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
        1.0,
    )

    if scale == 1.0:
        return frame.copy(), scale

    resized = cv2.resize(
        frame,
        (
            int(round(width * scale)),
            int(round(height * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


def save_roi_config(
    config_path: Path,
    source: Path,
    frame: np.ndarray,
    slots: list[dict[str, int]],
) -> None:
    """Save parking-slot rectangles to JSON."""

    height, width = frame.shape[:2]

    payload = {
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "source_used_for_setup": str(source),
        "frame_width": width,
        "frame_height": height,
        "slot_count": len(slots),
        "slots": slots,
    }

    config_path.write_text(
        json.dumps(
            payload,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def setup_rois(
    source: Path,
    config_path: Path,
) -> None:
    """
    Interactively select rectangular parking slots.

    Instructions are printed in the terminal instead of being drawn over
    the image. The OpenCV window uses autosize, so the image is not stretched.
    """

    frame = read_first_frame(source)

    display_frame, scale = resize_for_roi_selection(
        frame
    )

    print()
    print("=" * 78)
    print("PARKING ROI SETUP")
    print("=" * 78)
    print(f"Source image       : {source}")
    print(
        f"Original size      : "
        f"{frame.shape[1]} x {frame.shape[0]}"
    )
    print(
        f"Display size       : "
        f"{display_frame.shape[1]} x {display_frame.shape[0]}"
    )
    print("Draw one parking slot with the mouse.")
    print("Press ENTER or SPACE to confirm each slot.")
    print("Press ESC when all slots are completed.")
    print("Press C to cancel the current rectangle.")
    print("=" * 78)

    window_name = "Parking ROI Setup"

    # WINDOW_AUTOSIZE prevents OpenCV from stretching a small image
    # into a large blurry window.
    cv2.namedWindow(
        window_name,
        cv2.WINDOW_AUTOSIZE,
    )

    cv2.moveWindow(
        window_name,
        25,
        25,
    )

    cv2.imshow(
        window_name,
        display_frame,
    )

    cv2.waitKey(1)

    selected = cv2.selectROIs(
        window_name,
        display_frame,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyAllWindows()

    slots: list[dict[str, int]] = []

    for index, roi in enumerate(
        selected,
        start=1,
    ):
        x, y, width, height = [
            int(round(float(value) / scale))
            for value in roi
        ]

        if width <= 0 or height <= 0:
            continue

        slots.append(
            {
                "id": index,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        )

    if not slots:
        raise RuntimeError(
            "No parking slots were selected."
        )

    save_roi_config(
        config_path=config_path,
        source=source,
        frame=frame,
        slots=slots,
    )

    preview = frame.copy()

    for slot in slots:
        x1 = slot["x"]
        y1 = slot["y"]
        x2 = x1 + slot["width"]
        y2 = y1 + slot["height"]

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )

        cv2.putText(
            preview,
            f"S{slot['id']}",
            (x1 + 3, max(15, y1 + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    preview_path = (
        config_path.parent
        / f"{config_path.stem}_preview.jpg"
    )

    save_ok = cv2.imwrite(
        str(preview_path),
        preview,
    )

    if not save_ok:
        raise OSError(
            f"Could not save ROI preview:\n{preview_path}"
        )

    print()
    print("=" * 78)
    print("ROI SETUP COMPLETED")
    print("=" * 78)
    print(f"Selected slots     : {len(slots)}")
    print(f"ROI configuration  : {config_path}")
    print(f"ROI preview        : {preview_path}")
    print("=" * 78)


def load_slots(
    config_path: Path,
) -> list[dict[str, int]]:
    """Load and validate saved parking slots."""

    if not config_path.is_file():
        raise FileNotFoundError(
            "ROI configuration was not found.\n"
            f"Run the script once with --setup:\n{config_path}"
        )

    payload = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    slots = payload.get("slots", [])

    if not slots:
        raise ValueError(
            f"No slots are stored in:\n{config_path}"
        )

    return slots


def get_vehicle_class_ids(
    model: YOLO,
) -> list[int]:
    """Resolve COCO vehicle class IDs by class name."""

    names = model.names

    class_ids = [
        int(class_id)
        for class_id, class_name in names.items()
        if str(class_name).lower() in VEHICLE_CLASS_NAMES
    ]

    if not class_ids:
        raise ValueError(
            "The selected model does not contain standard vehicle classes."
        )

    return class_ids


def extract_vehicle_boxes(
    result: Any,
) -> list[dict[str, Any]]:
    """Extract vehicle bounding boxes and labels from one YOLO result."""

    vehicles: list[dict[str, Any]] = []

    boxes = getattr(result, "boxes", None)

    if boxes is None or len(boxes) == 0:
        return vehicles

    xyxy_values = boxes.xyxy.detach().cpu().tolist()
    class_values = boxes.cls.detach().cpu().tolist()
    confidence_values = boxes.conf.detach().cpu().tolist()

    for xyxy, class_value, confidence in zip(
        xyxy_values,
        class_values,
        confidence_values,
    ):
        class_id = int(class_value)
        class_name = str(
            result.names.get(class_id, class_id)
        ).lower()

        x1, y1, x2, y2 = [
            int(round(value))
            for value in xyxy
        ]

        vehicles.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(confidence),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    return vehicles


def rectangle_intersection_area(
    ax1: int,
    ay1: int,
    ax2: int,
    ay2: int,
    bx1: int,
    by1: int,
    bx2: int,
    by2: int,
) -> int:
    """Return the intersection area of two rectangles."""

    width = max(
        0,
        min(ax2, bx2) - max(ax1, bx1),
    )

    height = max(
        0,
        min(ay2, by2) - max(ay1, by1),
    )

    return width * height


def vehicle_occupies_slot(
    vehicle: dict[str, Any],
    slot: dict[str, int],
    minimum_overlap: float,
) -> bool:
    """
    Decide whether a vehicle occupies a parking slot.

    Primary rule:
        The bottom-center point of the vehicle box lies inside the slot ROI.

    Fallback rule:
        A sufficient part of the vehicle box overlaps the slot ROI.

    Bottom-center is usually more reliable than the geometric center for
    oblique camera views because it is closer to where the vehicle touches
    the road surface.
    """

    vx1 = vehicle["x1"]
    vy1 = vehicle["y1"]
    vx2 = vehicle["x2"]
    vy2 = vehicle["y2"]

    sx1 = slot["x"]
    sy1 = slot["y"]
    sx2 = sx1 + slot["width"]
    sy2 = sy1 + slot["height"]

    bottom_center_x = (vx1 + vx2) / 2.0
    bottom_center_y = float(vy2)

    bottom_center_inside = (
        sx1 <= bottom_center_x <= sx2
        and sy1 <= bottom_center_y <= sy2
    )

    if bottom_center_inside:
        return True

    vehicle_area = max(
        1,
        (vx2 - vx1) * (vy2 - vy1),
    )

    intersection_area = rectangle_intersection_area(
        vx1,
        vy1,
        vx2,
        vy2,
        sx1,
        sy1,
        sx2,
        sy2,
    )

    overlap_ratio = intersection_area / vehicle_area

    return overlap_ratio >= minimum_overlap


def calculate_slot_states(
    slots: list[dict[str, int]],
    vehicles: list[dict[str, Any]],
    minimum_overlap: float,
) -> list[dict[str, Any]]:
    """Calculate free/occupied state for every slot."""

    states: list[dict[str, Any]] = []

    for slot in slots:
        matching_vehicle = None

        for vehicle in vehicles:
            if vehicle_occupies_slot(
                vehicle=vehicle,
                slot=slot,
                minimum_overlap=minimum_overlap,
            ):
                matching_vehicle = vehicle
                break

        states.append(
            {
                **slot,
                "status": (
                    "occupied"
                    if matching_vehicle is not None
                    else "free"
                ),
                "vehicle": matching_vehicle,
            }
        )

    return states


def draw_results(
    frame: np.ndarray,
    slot_states: list[dict[str, Any]],
    vehicles: list[dict[str, Any]],
    inference_ms: float,
) -> np.ndarray:
    """
    Draw parking occupancy results.

    Default display:
        - Red ROI with OCCUPIED label
        - Green ROI with FREE label
        - No vehicle bounding boxes
        - Compact summary panel
    """

    output = frame.copy()

    # --------------------------------------------------------
    # Optional vehicle boxes
    # --------------------------------------------------------
    if SHOW_VEHICLE_BOXES:
        for vehicle in vehicles:
            cv2.rectangle(
                output,
                (vehicle["x1"], vehicle["y1"]),
                (vehicle["x2"], vehicle["y2"]),
                (255, 180, 0),
                2,
            )

            vehicle_label = (
                f"{vehicle['class_name']} "
                f"{vehicle['confidence']:.2f}"
            )

            cv2.putText(
                output,
                vehicle_label,
                (
                    vehicle["x1"],
                    max(20, vehicle["y1"] - 6),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 180, 0),
                1,
                cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # Draw parking-slot ROIs and status labels
    # --------------------------------------------------------
    for state in slot_states:
        x1 = int(state["x"])
        y1 = int(state["y"])
        x2 = x1 + int(state["width"])
        y2 = y1 + int(state["height"])

        occupied = state["status"] == "occupied"

        color = (
            (0, 0, 255)       # Red: occupied
            if occupied
            else (0, 200, 0)  # Green: free
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        if SHOW_STATUS_LABELS:
            status_label = (
                "OCCUPIED"
                if occupied
                else "FREE"
            )

            roi_width = max(1, x2 - x1)
            roi_height = max(1, y2 - y1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_thickness = 1

            # Start small and automatically reduce the text size
            # until the label fits inside the ROI width.
            font_scale = 0.38

            while font_scale > 0.20:
                (text_width, text_height), baseline = cv2.getTextSize(
                    status_label,
                    font,
                    font_scale,
                    font_thickness,
                )

                if text_width + 8 <= roi_width:
                    break

                font_scale -= 0.02

            # For extremely narrow slots, use a compact label.
            if text_width + 8 > roi_width:
                status_label = (
                    "OCC"
                    if occupied
                    else "FREE"
                )

                font_scale = 0.24

                (text_width, text_height), baseline = cv2.getTextSize(
                    status_label,
                    font,
                    font_scale,
                    font_thickness,
                )

            padding_x = 4
            padding_y = 3

            label_x1 = x1
            label_y1 = y1
            label_x2 = min(
                x2,
                x1 + text_width + padding_x * 2,
            )
            label_y2 = min(
                y2,
                y1 + text_height + baseline + padding_y * 2,
            )

            # Filled status-colored label background.
            cv2.rectangle(
                output,
                (label_x1, label_y1),
                (label_x2, label_y2),
                color,
                thickness=-1,
            )

            text_x = label_x1 + padding_x
            text_y = min(
                label_y2 - baseline - padding_y,
                y2 - 2,
            )

            cv2.putText(
                output,
                status_label,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # Count parking states
    # --------------------------------------------------------
    total_slots = len(slot_states)

    occupied_slots = sum(
        state["status"] == "occupied"
        for state in slot_states
    )

    free_slots = total_slots - occupied_slots

    # --------------------------------------------------------
    # Compact summary panel
    # --------------------------------------------------------
    lines = [
        f"Total    : {total_slots}",
        f"Free     : {free_slots}",
        f"Occupied : {occupied_slots}",
    ]

    if SHOW_INFERENCE_TIME:
        lines.append(
            f"Time     : {inference_ms:.1f} ms"
        )

    panel_width = 205
    panel_height = 18 + len(lines) * 22

    overlay = output.copy()

    cv2.rectangle(
        overlay,
        (8, 8),
        (8 + panel_width, 8 + panel_height),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.addWeighted(
        overlay,
        0.65,
        output,
        0.35,
        0,
        output,
    )

    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (18, 32 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def process_frame(
    model: YOLO,
    frame: np.ndarray,
    slots: list[dict[str, int]],
    vehicle_class_ids: list[int],
    args: argparse.Namespace,
    use_tracking: bool,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    """Detect vehicles and calculate parking occupancy for one frame."""

    start_time = time.perf_counter()

    if use_tracking:
        results = model.track(
            source=frame,
            persist=True,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=vehicle_class_ids,
            device=args.device,
            verbose=False,
        )
    else:
        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=vehicle_class_ids,
            device=args.device,
            verbose=False,
        )

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000.0

    result = results[0]
    vehicles = extract_vehicle_boxes(result)

    slot_states = calculate_slot_states(
        slots=slots,
        vehicles=vehicles,
        minimum_overlap=args.slot_overlap,
    )

    annotated = draw_results(
        frame=frame,
        slot_states=slot_states,
        vehicles=vehicles,
        inference_ms=elapsed_ms,
    )

    return annotated, slot_states, elapsed_ms


def save_image_summary(
    source: Path,
    output: Path,
    slot_states: list[dict[str, Any]],
    inference_ms: float,
) -> None:
    """Save image occupancy summary to JSON and CSV."""

    total_slots = len(slot_states)
    occupied_slots = sum(
        state["status"] == "occupied"
        for state in slot_states
    )
    free_slots = total_slots - occupied_slots

    summary = {
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "source": str(source),
        "output": str(output),
        "total_slots": total_slots,
        "free_slots": free_slots,
        "occupied_slots": occupied_slots,
        "inference_ms": inference_ms,
        "slots": [
            {
                "id": state["id"],
                "status": state["status"],
            }
            for state in slot_states
        ],
    }

    json_path = output.with_suffix(".json")
    csv_path = output.with_suffix(".csv")

    json_path.write_text(
        json.dumps(
            summary,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "slot_id",
                "status",
            ],
        )

        writer.writeheader()

        for state in slot_states:
            writer.writerow(
                {
                    "slot_id": state["id"],
                    "status": state["status"],
                }
            )


# ============================================================
# 5. IMAGE AND VIDEO PROCESSING
# ============================================================

def run_image(
    model: YOLO,
    args: argparse.Namespace,
    slots: list[dict[str, int]],
    vehicle_class_ids: list[int],
) -> None:
    """Process one image."""

    frame = cv2.imread(str(args.source))

    if frame is None:
        raise OSError(
            f"Could not read image:\n{args.source}"
        )

    annotated, slot_states, inference_ms = process_frame(
        model=model,
        frame=frame,
        slots=slots,
        vehicle_class_ids=vehicle_class_ids,
        args=args,
        use_tracking=False,
    )

    save_ok = cv2.imwrite(
        str(args.output),
        annotated,
    )

    if not save_ok:
        raise OSError(
            f"Could not save image:\n{args.output}"
        )

    save_image_summary(
        source=args.source,
        output=args.output,
        slot_states=slot_states,
        inference_ms=inference_ms,
    )

    if args.show:
        cv2.imshow(
            "Parking Occupancy Demo",
            annotated,
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    occupied_slots = sum(
        state["status"] == "occupied"
        for state in slot_states
    )

    print("=" * 78)
    print("IMAGE DEMO COMPLETED")
    print("=" * 78)
    print(f"Total slots        : {len(slot_states)}")
    print(f"Free slots         : {len(slot_states) - occupied_slots}")
    print(f"Occupied slots     : {occupied_slots}")
    print(f"Output image       : {args.output}")
    print("=" * 78)


def run_video(
    model: YOLO,
    args: argparse.Namespace,
    slots: list[dict[str, int]],
    vehicle_class_ids: list[int],
) -> None:
    """Process one fixed-camera video."""

    capture = cv2.VideoCapture(
        str(args.source)
    )

    if not capture.isOpened():
        raise OSError(
            f"Could not open video:\n{args.source}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise OSError(
            f"Could not create output video:\n{args.output}"
        )

    frame_index = 0
    last_states: list[dict[str, Any]] = []

    while True:
        ok, frame = capture.read()

        if not ok or frame is None:
            break

        annotated, slot_states, _ = process_frame(
            model=model,
            frame=frame,
            slots=slots,
            vehicle_class_ids=vehicle_class_ids,
            args=args,
            use_tracking=True,
        )

        writer.write(annotated)
        last_states = slot_states
        frame_index += 1

        if args.show:
           display = annotated

        max_display_width = 1200
        max_display_height = 700

        display_scale = min(
                max_display_width / width,
                max_display_height / height,
                1.0,
            )

        if display_scale < 1.0:
            display = cv2.resize(
            annotated,
        (
            int(width * display_scale),
            int(height * display_scale),
        ),
        interpolation=cv2.INTER_AREA,
        )

            cv2.imshow(
                "Parking Occupancy Video",
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

        if frame_index % 100 == 0:
            print(
                f"Processed frames: {frame_index}"
            )

    capture.release()
    writer.release()
    cv2.destroyAllWindows()

    occupied_slots = sum(
        state["status"] == "occupied"
        for state in last_states
    )

    print("=" * 78)
    print("VIDEO DEMO COMPLETED")
    print("=" * 78)
    print(f"Processed frames   : {frame_index}")
    print(f"Final total slots  : {len(last_states)}")
    print(
        f"Final free slots   : "
        f"{len(last_states) - occupied_slots}"
    )
    print(f"Final occupied     : {occupied_slots}")
    print(f"Output video       : {args.output}")
    print("=" * 78)


# ============================================================
# 6. MAIN
# ============================================================

def main() -> None:
    """Program entry point."""

    args = parse_arguments()

    try:
        validate_arguments(args)

        if args.setup:
            setup_rois(
                source=args.source,
                config_path=args.config,
            )
            return

        slots = load_slots(args.config)

        print("=" * 78)
        print("PARKING ROI DEMO")
        print("=" * 78)
        print(f"Source             : {args.source}")
        print(f"ROI configuration  : {args.config}")
        print(f"Parking slots      : {len(slots)}")
        print(f"Model              : {args.model}")
        print(f"CUDA available     : {torch.cuda.is_available()}")
        print(f"Device             : {args.device}")
        print(f"Output             : {args.output}")
        print("=" * 78)

        model = YOLO(args.model)

        vehicle_class_ids = get_vehicle_class_ids(
            model
        )

        if source_is_image(args.source):
            run_image(
                model=model,
                args=args,
                slots=slots,
                vehicle_class_ids=vehicle_class_ids,
            )
        else:
            run_video(
                model=model,
                args=args,
                slots=slots,
                vehicle_class_ids=vehicle_class_ids,
            )

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nDemo was stopped by the user.")
        raise SystemExit(130)

    except Exception as exc:
        cv2.destroyAllWindows()

        print()
        print("=" * 78)
        print("DEMO FAILED")
        print("=" * 78)
        print(f"{type(exc).__name__}: {exc}")
        print("=" * 78)

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()