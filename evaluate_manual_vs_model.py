"""Evaluate YOLO11 + ROI occupancy predictions against manual labels.

The script has three stages:

1. prepare  - sample video frames and save the model prediction per ROI.
2. label    - manually label every sampled ROI as free or occupied.
3. evaluate - calculate metrics and create report-ready outputs.

The ROI matching rule follows the parking application:
- a slot is occupied when the bottom-center point of a detected vehicle is
  inside the ROI; or
- the intersection area divided by the vehicle-box area is at least 0.20.
"""


# =========================================================
# 1. SUPPORTING LIBRARIES
# =========================================================

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 2. CONFIGURATION AND SUPPORTING FUNCTIONS
# =========================================================

VALID_LABELS = {"free", "occupied"}
VEHICLE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck"}


def parse_arguments():
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare YOLO11--ROI predictions with manual ground truth."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "label", "evaluate", "all"),
        default="all",
        help="Processing stage to run (default: all).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the original parking video.",
    )
    parser.add_argument(
        "--slots",
        type=Path,
        required=True,
        help="Path to the ROI JSON file.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="YOLO weight used by the app (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=10,
        help="Number of evenly spaced frames (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/manual_vs_model/cam1"),
        help="Directory for generated evaluation files.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--slot-overlap", type=float, default=0.20)
    parser.add_argument(
        "--device",
        default=None,
        help='Inference device, for example "0" or "cpu".',
    )
    return parser.parse_args()


def validate_inputs(args):
    """Check paths and argument values before processing."""
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")

    if not args.slots.is_file():
        raise FileNotFoundError(f"ROI JSON not found: {args.slots}")

    if args.sample_count < 1:
        raise ValueError("--sample-count must be at least 1.")

    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1.")

    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be between 0 and 1.")

    if not 0.0 <= args.slot_overlap <= 1.0:
        raise ValueError("--slot-overlap must be between 0 and 1.")


def load_slots(json_path):
    """Load and validate rectangular parking ROIs."""
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    slots = data.get("slots", data if isinstance(data, list) else [])

    if not slots:
        raise ValueError(f"No parking slots found in: {json_path}")

    required_keys = {"id", "x", "y", "width", "height"}

    for slot in slots:
        missing = required_keys.difference(slot)
        if missing:
            raise ValueError(
                f"ROI {slot.get('id', '?')} is missing: {sorted(missing)}"
            )

    return slots


def resolve_device(requested_device):
    """Use the requested device or select CUDA when available."""
    if requested_device is not None:
        return requested_device

    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def sample_frame_indices(total_frames, sample_count):
    """Select distinct frame indices without using the first/last frame."""
    if total_frames <= 0:
        raise ValueError("The video does not report a valid frame count.")

    actual_count = min(sample_count, total_frames)

    if actual_count == 1:
        return [total_frames // 2]

    indices = np.linspace(
        0,
        total_frames - 1,
        actual_count + 2,
        dtype=int,
    )[1:-1]

    return sorted(set(int(index) for index in indices))


def extract_vehicle_boxes(result):
    """Convert Ultralytics detections into plain dictionaries."""
    vehicles = []
    names = result.names

    if result.boxes is None:
        return vehicles

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

    for coordinates, confidence, class_id in zip(
        xyxy,
        confidences,
        class_ids,
    ):
        class_name = str(names[class_id]).lower()

        if class_name not in VEHICLE_CLASS_NAMES:
            continue

        x1, y1, x2, y2 = coordinates
        vehicles.append(
            {
                "x1": int(round(x1)),
                "y1": int(round(y1)),
                "x2": int(round(x2)),
                "y2": int(round(y2)),
                "confidence": float(confidence),
                "class_id": int(class_id),
                "class_name": class_name,
            }
        )

    return vehicles


def point_inside_rectangle(point_x, point_y, slot):
    """Return True when a point is inside a rectangular ROI."""
    slot_x1 = float(slot["x"])
    slot_y1 = float(slot["y"])
    slot_x2 = slot_x1 + float(slot["width"])
    slot_y2 = slot_y1 + float(slot["height"])

    return (
        slot_x1 <= point_x <= slot_x2
        and slot_y1 <= point_y <= slot_y2
    )


def vehicle_overlap_ratio(vehicle, slot):
    """Calculate intersection area divided by vehicle bounding-box area."""
    slot_x1 = float(slot["x"])
    slot_y1 = float(slot["y"])
    slot_x2 = slot_x1 + float(slot["width"])
    slot_y2 = slot_y1 + float(slot["height"])

    intersection_x1 = max(float(vehicle["x1"]), slot_x1)
    intersection_y1 = max(float(vehicle["y1"]), slot_y1)
    intersection_x2 = min(float(vehicle["x2"]), slot_x2)
    intersection_y2 = min(float(vehicle["y2"]), slot_y2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    vehicle_width = max(
        0.0,
        float(vehicle["x2"]) - float(vehicle["x1"]),
    )
    vehicle_height = max(
        0.0,
        float(vehicle["y2"]) - float(vehicle["y1"]),
    )
    vehicle_area = vehicle_width * vehicle_height

    if vehicle_area <= 0:
        return 0.0

    return intersection_area / vehicle_area


def calculate_slot_states(slots, vehicles, minimum_overlap):
    """Classify every ROI using the same rule described in the report."""
    states = []

    for slot in slots:
        best_vehicle = None
        best_overlap = 0.0
        matched_by_bottom_center = False

        for vehicle in vehicles:
            bottom_center_x = (vehicle["x1"] + vehicle["x2"]) / 2.0
            bottom_center_y = float(vehicle["y2"])
            inside = point_inside_rectangle(
                bottom_center_x,
                bottom_center_y,
                slot,
            )
            overlap = vehicle_overlap_ratio(vehicle, slot)

            if inside or overlap >= minimum_overlap:
                if best_vehicle is None or inside or overlap > best_overlap:
                    best_vehicle = vehicle
                    best_overlap = overlap
                    matched_by_bottom_center = inside

        states.append(
            {
                **slot,
                "status": (
                    "occupied" if best_vehicle is not None else "free"
                ),
                "vehicle": best_vehicle,
                "overlap_ratio": best_overlap,
                "bottom_center_match": matched_by_bottom_center,
            }
        )

    return states


def draw_prediction_frame(frame, states):
    """Draw model predictions for later error inspection."""
    import cv2

    output = frame.copy()

    for state in states:
        x1 = int(state["x"])
        y1 = int(state["y"])
        x2 = x1 + int(state["width"])
        y2 = y1 + int(state["height"])
        occupied = state["status"] == "occupied"
        color = (42, 42, 220) if occupied else (74, 184, 34)
        label = f"S{state['id']} {state['status'].upper()}"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

    return output


def save_dataframe(dataframe, csv_path):
    """Write CSV using UTF-8 and stable column ordering."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")


# =========================================================
# 3. PROCESSING FUNCTIONS
# =========================================================

def prepare_predictions(args, slots):
    """Sample frames, run YOLO11, and create the labeling CSV."""
    import cv2
    from ultralytics import YOLO

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "frames_raw"
    prediction_dir = args.output_dir / "frames_model_predictions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video))

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_indices = sample_frame_indices(total_frames, args.sample_count)

    model = YOLO(args.model)
    device = resolve_device(args.device)
    records = []

    print(
        f"Preparing {len(frame_indices)} frames from {args.video.name} "
        f"with {len(slots)} ROIs..."
    )

    for sequence, frame_index in enumerate(frame_indices, start=1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()

        if not success or frame is None:
            print(f"Warning: cannot read frame {frame_index}; skipped.")
            continue

        result = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device,
            verbose=False,
        )[0]
        vehicles = extract_vehicle_boxes(result)
        states = calculate_slot_states(
            slots,
            vehicles,
            args.slot_overlap,
        )

        frame_name = f"sample_{sequence:02d}_frame_{frame_index:06d}.jpg"
        raw_path = raw_dir / frame_name
        prediction_path = prediction_dir / frame_name
        cv2.imwrite(str(raw_path), frame)
        cv2.imwrite(
            str(prediction_path),
            draw_prediction_frame(frame, states),
        )

        timestamp_seconds = frame_index / fps if fps > 0 else np.nan

        for state in states:
            records.append(
                {
                    "source": args.video.stem,
                    "sample_number": sequence,
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "frame_path": str(raw_path.resolve()),
                    "roi_id": state["id"],
                    "model_label": state["status"],
                    "manual_label": "",
                }
            )

    capture.release()

    if not records:
        raise RuntimeError("No video frames were prepared.")

    labels_path = args.output_dir / "manual_labels.csv"
    dataframe = pd.DataFrame(records)
    save_dataframe(dataframe, labels_path)

    metadata = {
        "video": str(args.video.resolve()),
        "roi_config": str(args.slots.resolve()),
        "model": args.model,
        "sample_count_requested": args.sample_count,
        "sample_count_prepared": int(dataframe["sample_number"].nunique()),
        "roi_count": len(slots),
        "roi_observations": len(dataframe),
        "imgsz": args.imgsz,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "minimum_vehicle_overlap": args.slot_overlap,
        "device": device,
    }

    with (args.output_dir / "experiment_config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print(f"Prepared ROI observations: {len(dataframe)}")
    print(f"Manual-label file: {labels_path.resolve()}")


def draw_manual_labeling_view(frame, slot, row_number, total_rows):
    """Highlight one ROI without revealing its model prediction."""
    import cv2

    output = frame.copy()
    x1 = int(slot["x"])
    y1 = int(slot["y"])
    x2 = x1 + int(slot["width"])
    y2 = y1 + int(slot["height"])

    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], 85), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.80, output, 0.20, 0, output)
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 230, 255), 4)

    title = (
        f"Manual ground truth  |  ROI S{slot['id']}  |  "
        f"{row_number}/{total_rows}"
    )
    instructions = (
        "F or 0 = FREE    O or 1 = OCCUPIED    "
        "B = BACK    Q/ESC = SAVE AND EXIT"
    )

    cv2.putText(
        output,
        title,
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        instructions,
        (18, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 220, 230),
        1,
        cv2.LINE_AA,
    )

    return output


def label_ground_truth(args, slots):
    """Interactively assign free/occupied ground-truth labels."""
    import cv2

    labels_path = args.output_dir / "manual_labels.csv"

    if not labels_path.is_file():
        raise FileNotFoundError(
            "manual_labels.csv was not found. Run --stage prepare first."
        )

    dataframe = pd.read_csv(
        labels_path,
        dtype={"manual_label": "string"},
        keep_default_na=False,
    )
    slot_lookup = {str(slot["id"]): slot for slot in slots}
    valid_rows = dataframe["manual_label"].isin(VALID_LABELS)
    current_index = int(np.argmax(~valid_rows.to_numpy())) if not valid_rows.all() else 0
    window_name = "Manual ROI Ground Truth"

    print("Manual labeling started. Model predictions are hidden.")

    while 0 <= current_index < len(dataframe):
        row = dataframe.iloc[current_index]
        slot = slot_lookup.get(str(row["roi_id"]))

        if slot is None:
            raise KeyError(f"ROI {row['roi_id']} is missing from the JSON file.")

        frame = cv2.imread(str(row["frame_path"]))

        if frame is None:
            raise FileNotFoundError(f"Cannot read frame: {row['frame_path']}")

        view = draw_manual_labeling_view(
            frame,
            slot,
            current_index + 1,
            len(dataframe),
        )
        cv2.imshow(window_name, view)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("f"), ord("F"), ord("0")):
            dataframe.at[current_index, "manual_label"] = "free"
            current_index += 1
        elif key in (ord("o"), ord("O"), ord("1")):
            dataframe.at[current_index, "manual_label"] = "occupied"
            current_index += 1
        elif key in (ord("b"), ord("B"), 8):
            current_index = max(0, current_index - 1)
        elif key in (ord("q"), ord("Q"), 27):
            save_dataframe(dataframe, labels_path)
            cv2.destroyAllWindows()
            print("Progress saved. Labeling can be resumed later.")
            return False

        save_dataframe(dataframe, labels_path)

    cv2.destroyAllWindows()
    print(f"Manual labeling complete: {len(dataframe)} ROI observations.")
    return True


def calculate_binary_metrics(dataframe):
    """Calculate occupied-positive binary classification metrics."""
    manual = dataframe["manual_label"]
    model = dataframe["model_label"]

    tp = int(((manual == "occupied") & (model == "occupied")).sum())
    tn = int(((manual == "free") & (model == "free")).sum())
    fp = int(((manual == "free") & (model == "occupied")).sum())
    fn = int(((manual == "occupied") & (model == "free")).sum())
    total = tp + tn + fp + fn
    correct = tp + tn

    accuracy = correct / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "roi_observations": total,
        "correct": correct,
        "incorrect": fp + fn,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def save_confusion_matrix(metrics, output_path):
    """Create a report-ready confusion matrix image."""
    import matplotlib.pyplot as plt

    matrix = np.array(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ]
    )

    figure, axis = plt.subplots(figsize=(5.2, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set_xticks([0, 1], labels=["Free", "Occupied"])
    axis.set_yticks([0, 1], labels=["Free", "Occupied"])
    axis.set_xlabel("Model prediction")
    axis.set_ylabel("Manual ground truth")
    axis.set_title("YOLO11--ROI Occupancy Confusion Matrix")

    threshold = matrix.max() / 2.0 if matrix.size else 0

    for row_index in range(2):
        for column_index in range(2):
            value = int(matrix[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=13,
                color="white" if value > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_latex_table(source, metrics, output_path):
    """Write a compact LaTeX table row for the report."""
    latex = rf"""\begin{{table}}[!htbp]
    \centering
    \caption{{Comparison between manual ground-truth labels and
    YOLO11--ROI predictions on the university parking video.}}
    \label{{tab:manual_model_comparison}}
    \begin{{tabular}}{{lrrrr}}
        \hline
        \textbf{{Source}} &
        \textbf{{ROI observations}} &
        \textbf{{Correct}} &
        \textbf{{Incorrect}} &
        \textbf{{Accuracy (\%)}} \\
        \hline
        {source} &
        {metrics['roi_observations']} &
        {metrics['correct']} &
        {metrics['incorrect']} &
        {metrics['accuracy'] * 100:.2f} \\
        \hline
    \end{{tabular}}
\end{{table}}
"""
    output_path.write_text(latex, encoding="utf-8")


def evaluate_predictions(args):
    """Compare completed manual labels with model predictions."""
    labels_path = args.output_dir / "manual_labels.csv"

    if not labels_path.is_file():
        raise FileNotFoundError(
            "manual_labels.csv was not found. Run --stage prepare first."
        )

    dataframe = pd.read_csv(labels_path, keep_default_na=False)
    invalid_manual = ~dataframe["manual_label"].isin(VALID_LABELS)
    invalid_model = ~dataframe["model_label"].isin(VALID_LABELS)

    if invalid_manual.any():
        missing_count = int(invalid_manual.sum())
        raise ValueError(
            f"Manual labeling is incomplete: {missing_count} rows remain."
        )

    if invalid_model.any():
        raise ValueError("model_label contains unsupported values.")

    dataframe["correct"] = (
        dataframe["manual_label"] == dataframe["model_label"]
    )
    metrics = calculate_binary_metrics(dataframe)
    source = str(dataframe["source"].iloc[0])

    per_frame_rows = []
    for (sample_number, frame_index), group in dataframe.groupby(
        ["sample_number", "frame_index"],
        sort=True,
    ):
        frame_metrics = calculate_binary_metrics(group)
        per_frame_rows.append(
            {
                "sample_number": sample_number,
                "frame_index": frame_index,
                "roi_observations": frame_metrics["roi_observations"],
                "correct": frame_metrics["correct"],
                "incorrect": frame_metrics["incorrect"],
                "accuracy_percent": frame_metrics["accuracy"] * 100.0,
            }
        )

    summary_row = {
        "source": source,
        "roi_observations": metrics["roi_observations"],
        "correct": metrics["correct"],
        "incorrect": metrics["incorrect"],
        "accuracy_percent": metrics["accuracy"] * 100.0,
        "precision_percent": metrics["precision"] * 100.0,
        "recall_percent": metrics["recall"] * 100.0,
        "f1_score_percent": metrics["f1_score"] * 100.0,
        "true_positive": metrics["true_positive"],
        "true_negative": metrics["true_negative"],
        "false_positive": metrics["false_positive"],
        "false_negative": metrics["false_negative"],
    }

    save_dataframe(
        dataframe,
        args.output_dir / "manual_vs_model_detailed.csv",
    )
    save_dataframe(
        pd.DataFrame(per_frame_rows),
        args.output_dir / "per_frame_accuracy.csv",
    )
    save_dataframe(
        pd.DataFrame([summary_row]),
        args.output_dir / "evaluation_summary.csv",
    )
    save_confusion_matrix(
        metrics,
        args.output_dir / "confusion_matrix.png",
    )
    save_latex_table(
        source,
        metrics,
        args.output_dir / "manual_vs_model_table.tex",
    )

    with (args.output_dir / "evaluation_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary_row, file, indent=2, ensure_ascii=False)

    print("\nEvaluation complete")
    print(f"ROI observations : {metrics['roi_observations']}")
    print(f"Correct          : {metrics['correct']}")
    print(f"Incorrect        : {metrics['incorrect']}")
    print(f"Accuracy         : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision        : {metrics['precision'] * 100:.2f}%")
    print(f"Recall           : {metrics['recall'] * 100:.2f}%")
    print(f"F1-score         : {metrics['f1_score'] * 100:.2f}%")
    print(f"Results directory: {args.output_dir.resolve()}")


# =========================================================
# 4. MAIN FUNCTION
# =========================================================

def main():
    args = parse_arguments()
    validate_inputs(args)
    slots = load_slots(args.slots)

    if args.stage in ("prepare", "all"):
        prepare_predictions(args, slots)

    labeling_complete = True

    if args.stage in ("label", "all"):
        labeling_complete = label_ground_truth(args, slots)

    if args.stage == "evaluate" or (
        args.stage == "all" and labeling_complete
    ):
        evaluate_predictions(args)


if __name__ == "__main__":
    main()
