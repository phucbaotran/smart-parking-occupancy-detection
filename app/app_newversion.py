# =========================================================
# Parking Availability Dashboard - Polished PyQt App
# =========================================================
# Basic GUI only:
# - Left panel: RUN / STOP / SETUP
# - Main screen: 3 camera/video panels
# - Bottom/left panel: Total / Free / Occupied
#
# Current version:
# - RUN: choose up to 3 image/video files and display/play them
# - STOP: stop all videos
# - SETUP: placeholder message for ROI setup
#
# Later integration:
# - Replace the raw frame display in CameraPanel.update_frame()
#   with your YOLO11 + ROI processing function.
# =========================================================

import re
import sys
from datetime import datetime
from time import perf_counter
from types import SimpleNamespace
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# main.py is expected to be stored in PROJECT_ROOT/app/main.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.yolo_detect.demo_parking_roi_v3 import (  # noqa: E402
    calculate_slot_states,
    extract_vehicle_boxes,
    get_vehicle_class_ids,
    load_slots,
    setup_rois,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

ROI_CONFIG_DIR = PROJECT_ROOT / "demo" / "config"

# Change this to "yolo11s.pt" only when that is the weight used in the report.
YOLO_MODEL = "yolo11n.pt"


def draw_clean_parking_overlay(frame, states):
    """Draw parking results without the duplicate black summary panel."""
    output = frame.copy()

    matched_vehicles = {
        id(state["vehicle"]): state["vehicle"]
        for state in states
        if state.get("vehicle") is not None
    }

    # Only show vehicles that actually match a parking space. This keeps
    # unrelated detections on roads and sidewalks out of the dashboard.
    for vehicle in matched_vehicles.values():
        cv2.rectangle(
            output,
            (vehicle["x1"], vehicle["y1"]),
            (vehicle["x2"], vehicle["y2"]),
            (235, 166, 54),
            2,
        )

        vehicle_text = (
            f"{vehicle['class_name']} "
            f"{vehicle['confidence']:.2f}"
        )
        cv2.putText(
            output,
            vehicle_text,
            (vehicle["x1"], max(20, vehicle["y1"] - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 166, 54),
            1,
            cv2.LINE_AA,
        )

    for state in states:
        x1 = int(state["x"])
        y1 = int(state["y"])
        x2 = x1 + int(state["width"])
        y2 = y1 + int(state["height"])

        occupied = state["status"] == "occupied"
        color = (42, 42, 220) if occupied else (74, 184, 34)
        # Keep the ROI label short so it stays readable after the original
        # camera frame is scaled down to fit the dashboard panel.
        status_text = "OCCUPIED" if occupied else "FREE"
        label = f"S{state['id']}  {status_text}"

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        roi_width = max(1, x2 - x1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame_short_side = min(output.shape[:2])
        font_scale = max(
            0.48,
            min(0.68, frame_short_side / 1600.0),
        )
        thickness = 1

        while font_scale > 0.30:
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness,
            )
            if text_width + 10 <= roi_width:
                break
            font_scale -= 0.03

        if text_width + 10 > roi_width:
            label = f"S{state['id']}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                font,
                0.30,
                thickness,
            )
            font_scale = 0.30

        label_bottom = min(
            y2,
            y1 + text_height + baseline + 8,
        )
        cv2.rectangle(
            output,
            (x1, y1),
            (min(x2, x1 + text_width + 10), label_bottom),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 5, label_bottom - baseline - 3),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return output


def draw_camera_status_overlay(
    frame,
    available,
    occupied,
    status,
    fps,
    updated_at,
):
    """Draw large per-camera live statistics inside the video frame."""
    output = frame.copy()
    frame_height, frame_width = output.shape[:2]

    scale = max(
        0.80,
        min(1.50, frame_width / 1280.0),
    )
    margin = max(10, int(12 * scale))
    gap = max(6, int(7 * scale))
    status_height = max(34, int(38 * scale))
    card_height = max(78, int(88 * scale))
    panel_width = min(
        frame_width - (2 * margin),
        int(720 * scale),
    )

    if panel_width < 240:
        return output

    x1 = margin
    y1 = margin
    x2 = x1 + panel_width
    status_y2 = y1 + status_height
    cards_y1 = status_y2 + gap
    cards_y2 = min(
        frame_height - margin,
        cards_y1 + card_height,
    )

    card_width = (panel_width - gap) // 2
    available_x2 = x1 + card_width
    occupied_x1 = available_x2 + gap

    overlay = output.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, status_y2),
        (42, 23, 15),
        -1,
    )
    cv2.rectangle(
        overlay,
        (x1, cards_y1),
        (available_x2, cards_y2),
        (52, 101, 22),
        -1,
    )
    cv2.rectangle(
        overlay,
        (occupied_x1, cards_y1),
        (x2, cards_y2),
        (27, 27, 153),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.84,
        output,
        0.16,
        0,
        output,
    )

    if status == "LIVE":
        status_color = (74, 222, 128)
        fps_text = f"{fps:.1f} FPS" if fps > 0 else "-- FPS"
        status_text = f"LIVE  |  {fps_text}  |  {updated_at}"
    elif status == "IMAGE":
        status_color = (235, 183, 82)
        status_text = f"IMAGE  |  UPDATED {updated_at}"
    else:
        status_color = (148, 163, 184)
        status_text = f"STOPPED  |  LAST UPDATE {updated_at}"

    dot_x = x1 + max(12, int(16 * scale))
    dot_y = y1 + (status_height // 2)
    cv2.circle(
        output,
        (dot_x, dot_y),
        max(4, int(5 * scale)),
        status_color,
        -1,
        cv2.LINE_AA,
    )

    status_font_scale = max(0.80, 0.70 * scale)
    card_label_scale = max(0.78, 0.62 * scale)
    card_value_scale = max(1.45, 1.18 * scale)
    small_thickness = max(1, int(round(scale)))
    value_thickness = max(2, int(round(2 * scale)))

    cv2.putText(
        output,
        status_text,
        (
            dot_x + max(12, int(13 * scale)),
            y1 + int(status_height * 0.70),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        status_font_scale,
        (241, 245, 249),
        small_thickness,
        cv2.LINE_AA,
    )

    text_margin = max(12, int(14 * scale))
    label_y = cards_y1 + max(25, int(28 * scale))
    value_y = cards_y2 - max(12, int(12 * scale))

    cv2.putText(
        output,
        "AVAILABLE",
        (x1 + text_margin, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        card_label_scale,
        (187, 247, 208),
        small_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        str(available),
        (x1 + text_margin, value_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        card_value_scale,
        (255, 255, 255),
        value_thickness,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        "OCCUPIED",
        (occupied_x1 + text_margin, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        card_label_scale,
        (191, 191, 254),
        small_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        str(occupied),
        (occupied_x1 + text_margin, value_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        card_value_scale,
        (255, 255, 255),
        value_thickness,
        cv2.LINE_AA,
    )

    return output


# =========================================================
# 1. YOLO11 + ROI SERVICE
# =========================================================

class ParkingDetectionService:
    """Load YOLO once and reuse it for all camera panels."""

    def __init__(self):
        self.model = None
        self.vehicle_class_ids = []
        self.args = SimpleNamespace(
            imgsz=640,
            conf=0.35,
            iou=0.70,
            slot_overlap=0.20,
            device="0" if torch.cuda.is_available() else "cpu",
        )

    def ensure_model(self):
        """Load the pretrained detector only when it is first needed."""
        if self.model is None:
            self.model = YOLO(YOLO_MODEL)
            self.vehicle_class_ids = get_vehicle_class_ids(self.model)

    def process(self, frame, slots):
        """Return annotated frame and Total/Free/Occupied counts."""
        self.ensure_model()

        results = self.model.predict(
            source=frame,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            iou=self.args.iou,
            classes=self.vehicle_class_ids,
            device=self.args.device,
            verbose=False,
        )

        vehicles = extract_vehicle_boxes(results[0])
        states = calculate_slot_states(
            slots=slots,
            vehicles=vehicles,
            minimum_overlap=self.args.slot_overlap,
        )

        annotated = draw_clean_parking_overlay(frame, states)

        occupied = sum(
            state["status"] == "occupied"
            for state in states
        )
        total = len(states)
        free = total - occupied

        return annotated, total, free, occupied


# =========================================================
# 2. CAMERA PANEL
# =========================================================

class CameraPanel(QFrame):
    """One display panel for image/video input."""

    def __init__(
        self,
        camera_name: str,
        detection_service: ParkingDetectionService,
    ):
        super().__init__()

        self.camera_name = camera_name
        self.detection_service = detection_service
        self.source_path = None
        self.config_path = None
        self.slots = []
        self.capture = None
        self.is_video = False
        self.source_kind = None
        self.is_live = False
        self.current_frame = None
        self.last_base_frame = None
        self.last_display_frame = None
        self.total_slots = 0
        self.free_slots = 0
        self.occupied_slots = 0
        self.current_fps = 0.0
        self.last_frame_timestamp = None
        self.last_update_text = "--:--:--"

        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("CameraPanel")

        header = QFrame()
        header.setObjectName("CameraHeader")

        self.title_label = QLabel(camera_name)
        self.title_label.setObjectName("CameraTitle")
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.stats_label = QLabel("● OFFLINE")
        self.stats_label.setObjectName("CameraStats")
        self.stats_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(10, 5, 10, 5)
        header_layout.setSpacing(8)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.stats_label)
        header.setLayout(header_layout)

        self.video_label = QLabel("No Signal")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 220)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setObjectName("VideoArea")

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)
        layout.addWidget(header)
        layout.addWidget(self.video_label)
        self.setLayout(layout)

    def load_source(self, file_path: str):
        """Load image or video source."""
        self.stop()

        path = Path(file_path)
        self.source_path = str(path)
        self.setToolTip(str(path))
        self.config_path = (
            ROI_CONFIG_DIR / f"{path.stem}_slots.json"
        )
        self.load_roi_config()
        suffix = path.suffix.lower()

        if suffix in IMAGE_EXTENSIONS:
            self.is_video = False
            self.source_kind = "image"
            frame = cv2.imread(str(path))
            if frame is None:
                self.show_message("Cannot load image")
                return
            self.current_frame = frame
            self.process_and_display(frame)

        elif suffix in VIDEO_EXTENSIONS:
            self.is_video = True
            self.source_kind = "video"
            self.capture = cv2.VideoCapture(str(path))
            if not self.capture.isOpened():
                self.show_message("Cannot open video")
                self.capture = None
                return
            self.show_message(
                "Ready" if self.slots else "ROI setup required"
            )
            self.update_title()

        else:
            self.source_kind = None
            self.show_message("Unsupported file")

    def load_roi_config(self):
        """Load the ROI JSON that belongs to the selected source."""
        self.slots = []

        if self.config_path is None or not self.config_path.is_file():
            self.update_title()
            return

        self.slots = load_slots(self.config_path)
        self.update_title()

    def update_title(self):
        if self.source_path is None:
            status = "● OFFLINE"
            color = "#94A3B8"

        elif not self.slots:
            status = "● SETUP REQUIRED"
            color = "#D97706"

        elif self.source_kind == "video" and self.is_live:
            status = (
                f"● LIVE · {self.current_fps:.1f} FPS · "
                f"{self.last_update_text}"
            )
            color = "#15803D"

        elif self.source_kind == "video":
            if self.capture is not None:
                status = f"● READY · {len(self.slots)} spaces"
                color = "#2563EB"
            else:
                status = "● STOPPED"
                color = "#64748B"

        elif self.total_slots > 0:
            status = f"● IMAGE · updated {self.last_update_text}"
            color = "#2563EB"

        else:
            status = f"● {len(self.slots)} spaces · ready"
            color = "#2563EB"

        self.title_label.setText(self.camera_name.upper())
        self.stats_label.setText(status)
        self.stats_label.setStyleSheet(
            f"color: {color}; background: transparent;"
        )

    def process_and_display(self, frame):
        """Run YOLO11 + ROI when a valid configuration is available."""
        if not self.slots:
            self.total_slots = 0
            self.free_slots = 0
            self.occupied_slots = 0
            if self.is_video:
                self.update_realtime_metrics()
            self.update_title()
            self.display_frame(frame)
            return self.get_counts()

        annotated, total, free, occupied = (
            self.detection_service.process(frame, self.slots)
        )

        self.total_slots = total
        self.free_slots = free
        self.occupied_slots = occupied

        if self.is_video:
            self.update_realtime_metrics()
            display_status = "LIVE"
        else:
            self.is_live = False
            self.current_fps = 0.0
            self.last_update_text = datetime.now().strftime("%H:%M:%S")
            display_status = "IMAGE"

        self.last_base_frame = annotated.copy()
        dashboard_frame = draw_camera_status_overlay(
            frame=annotated,
            available=self.free_slots,
            occupied=self.occupied_slots,
            status=display_status,
            fps=self.current_fps,
            updated_at=self.last_update_text,
        )

        self.update_title()
        self.display_frame(dashboard_frame)

        return self.get_counts()

    def update_realtime_metrics(self):
        """Measure the real display update rate for this camera."""
        current_timestamp = perf_counter()

        if self.last_frame_timestamp is not None:
            elapsed = current_timestamp - self.last_frame_timestamp

            if elapsed > 0:
                instantaneous_fps = min(120.0, 1.0 / elapsed)

                if self.current_fps <= 0:
                    self.current_fps = instantaneous_fps
                else:
                    self.current_fps = (
                        0.80 * self.current_fps
                        + 0.20 * instantaneous_fps
                    )

        self.last_frame_timestamp = current_timestamp
        self.last_update_text = datetime.now().strftime("%H:%M:%S")
        self.is_live = True

    def get_counts(self):
        return (
            self.total_slots,
            self.free_slots,
            self.occupied_slots,
        )

    def update_frame(self):
        """Read and display one video frame."""
        if not self.is_video or self.capture is None:
            return

        ret, frame = self.capture.read()

        # Loop video automatically for demo purpose
        if not ret:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.capture.read()

        if not ret or frame is None:
            self.show_message("No Signal")
            return

        self.current_frame = frame

        return self.process_and_display(frame)

    def display_frame(self, frame):
        """Convert OpenCV BGR frame to QPixmap and show on QLabel."""
        self.last_display_frame = frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        q_img = QImage(
            rgb_frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.video_label.setPixmap(scaled_pixmap)

    def show_message(self, message: str):
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText(message)

    def stop(self):
        """Release video source."""
        if self.capture is not None:
            self.capture.release()

        self.capture = None
        self.is_video = False
        self.is_live = False
        self.current_fps = 0.0
        self.last_frame_timestamp = None
        self.update_title()

        if (
            self.source_kind == "video"
            and self.last_base_frame is not None
            and self.slots
        ):
            stopped_frame = draw_camera_status_overlay(
                frame=self.last_base_frame,
                available=self.free_slots,
                occupied=self.occupied_slots,
                status="STOPPED",
                fps=0.0,
                updated_at=self.last_update_text,
            )
            self.display_frame(stopped_frame)

    def clear(self):
        self.stop()
        self.source_path = None
        self.config_path = None
        self.slots = []
        self.source_kind = None
        self.is_live = False
        self.current_frame = None
        self.last_base_frame = None
        self.last_display_frame = None
        self.setToolTip("")
        self.total_slots = 0
        self.free_slots = 0
        self.occupied_slots = 0
        self.current_fps = 0.0
        self.last_frame_timestamp = None
        self.last_update_text = "--:--:--"
        self.update_title()
        self.show_message("No Signal")


# =========================================================
# 3. MAIN WINDOW
# =========================================================

class MainWindow(QMainWindow):
    """Main GUI window."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Parking Availability Monitor")
        self.setMinimumSize(1280, 760)

        self.camera_panels = []
        self.detection_service = ParkingDetectionService()
        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.update_video_frames)

        self.total_slots = 0
        self.free_slots = 0
        self.occupied_slots = 0
        self.monitoring_active = None

        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(16, 14, 16, 12)
        main_layout.setSpacing(12)

        top_bar = QFrame()
        top_bar.setObjectName("TopBar")

        title = QLabel("PARKING AVAILABILITY")
        title.setObjectName("AppTitle")

        subtitle = QLabel(
            "Multi-camera occupancy monitoring dashboard"
        )
        subtitle.setObjectName("AppSubtitle")

        brand_layout = QVBoxLayout()
        brand_layout.setContentsMargins(0, 0, 0, 0)
        brand_layout.setSpacing(1)
        brand_layout.addWidget(title)
        brand_layout.addWidget(subtitle)

        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(18, 10, 18, 10)
        top_layout.addLayout(brand_layout)
        top_layout.addStretch()
        top_bar.setLayout(top_layout)

        main_layout.addWidget(top_bar)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(12)

        left_panel = self.create_left_panel()
        camera_area = self.create_camera_area()

        body_layout.addWidget(left_panel, 1)
        body_layout.addWidget(camera_area, 5)

        main_layout.addLayout(body_layout)

        self.footer_label = QLabel()
        self.footer_label.setObjectName("TechnicalFooter")
        self.footer_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        main_layout.addWidget(self.footer_label)
        self.update_footer()

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def create_left_panel(self):
        panel = QFrame()
        panel.setObjectName("LeftPanel")
        panel.setFixedWidth(250)

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        controls_title = QLabel("CONTROL CENTER")
        controls_title.setObjectName("SectionTitle")
        layout.addWidget(controls_title)

        self.run_button = QPushButton("START MONITORING")
        self.stop_button = QPushButton("STOP")
        self.setup_button = QPushButton("SETUP SPACES")

        self.run_button.setObjectName("RunButton")
        self.stop_button.setObjectName("StopButton")
        self.setup_button.setObjectName("SetupButton")

        for button in [self.run_button, self.stop_button, self.setup_button]:
            button.setMinimumHeight(46)
            button.setCursor(Qt.PointingHandCursor)

        self.run_button.clicked.connect(self.run_app)
        self.stop_button.clicked.connect(self.stop_app)
        self.setup_button.clicked.connect(self.open_setup)

        layout.addWidget(self.run_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.setup_button)

        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumHeight(62)

        layout.addWidget(self.status_label)
        layout.addSpacing(14)

        self.set_control_state(False)
        self.update_system_status(False)

        stats_title = QLabel("LIVE OVERVIEW")
        stats_title.setObjectName("SectionTitle")
        layout.addWidget(stats_title)

        self.total_label = self.create_stat_label(
            "TOTAL", "0", "neutral"
        )
        self.free_label = self.create_stat_label(
            "AVAILABLE", "0", "available"
        )
        self.occupied_label = self.create_stat_label(
            "OCCUPIED", "0", "occupied"
        )
        self.rate_label = self.create_stat_label(
            "OCCUPANCY", "0.0%", "rate"
        )

        stats_grid = QGridLayout()
        stats_grid.setContentsMargins(0, 0, 0, 0)
        stats_grid.setHorizontalSpacing(8)
        stats_grid.setVerticalSpacing(8)
        stats_grid.addWidget(self.total_label, 0, 0)
        stats_grid.addWidget(self.free_label, 0, 1)
        stats_grid.addWidget(self.occupied_label, 1, 0)
        stats_grid.addWidget(self.rate_label, 1, 1)
        layout.addLayout(stats_grid)
        layout.addStretch()

        panel.setLayout(layout)
        return panel

    def create_stat_label(
        self,
        name: str,
        value: str,
        card_type: str,
    ):
        label = QLabel(f"{name}\n{value}")
        label.setAlignment(Qt.AlignCenter)
        label.setObjectName("StatBox")
        label.setProperty("cardType", card_type)
        label.setMinimumHeight(78)
        return label

    def create_camera_area(self):
        container = QFrame()
        container.setObjectName("CameraArea")

        layout = QGridLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 3-camera basic layout: one large panel + two smaller panels
        cam1 = CameraPanel("Camera 1", self.detection_service)
        cam2 = CameraPanel("Camera 2", self.detection_service)
        cam3 = CameraPanel("Camera 3", self.detection_service)

        self.camera_panels = [cam1, cam2, cam3]

        # Camera 2: video dọc, nằm bên trái và chiếm hai hàng
        layout.addWidget(cam2, 0, 0, 2, 1)

        # Camera 1 và Camera 3: video ngang, xếp bên phải
        layout.addWidget(cam1, 0, 1, 1, 2)
        layout.addWidget(cam3, 1, 1, 1, 2)

        # Điều chỉnh tỷ lệ kích thước
        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(2, 2)

        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)

        container.setLayout(layout)
        return container

    def assign_sources_to_cameras(self, file_paths):
        """Map cam1/cam2/cam3 filenames to the matching dashboard panel."""
        assigned = [None, None, None]
        unmatched = []

        for file_path in file_paths:
            stem = Path(file_path).stem.lower()
            match = re.search(r"cam(?:era)?[_\s-]*([123])", stem)

            if match:
                camera_index = int(match.group(1)) - 1
                if assigned[camera_index] is None:
                    assigned[camera_index] = file_path
                    continue

            unmatched.append(file_path)

        empty_indices = [
            index
            for index, source in enumerate(assigned)
            if source is None
        ]

        for index, file_path in zip(empty_indices, unmatched):
            assigned[index] = file_path

        return assigned

    def run_app(self):
        """Choose up to three sources and start YOLO11 + ROI analysis."""
        self.timer.stop()

        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select up to 3 image/video files",
            "",
            "Media Files (*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv *.wmv)",
        )

        if not file_paths:
            return

        file_paths = file_paths[:3]
        assigned_sources = self.assign_sources_to_cameras(file_paths)

        for panel in self.camera_panels:
            panel.clear()

        for index, file_path in enumerate(assigned_sources):
            if file_path is None:
                continue

            try:
                self.camera_panels[index].load_source(file_path)
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Cannot load source",
                    f"{Path(file_path).name}\n\n"
                    f"{type(exc).__name__}: {exc}",
                )

        missing_roi = [
            Path(panel.source_path).name
            for panel in self.camera_panels
            if panel.source_path and not panel.slots
        ]

        has_video = any(
            panel.is_video
            for panel in self.camera_panels
        )

        if has_video:
            self.timer.start(30)

        self.collect_stats()

        active_sources = sum(
            panel.source_path is not None
            for panel in self.camera_panels
        )

        if active_sources > 0:
            self.update_system_status(True)
            self.set_control_state(True)

        self.update_footer()

        if missing_roi:
            QMessageBox.information(
                self,
                "ROI setup required",
                "These sources do not have an ROI configuration yet:\n\n"
                + "\n".join(f"• {name}" for name in missing_roi)
                + "\n\nSelect SETUP and draw the parking slots.",
            )

    def stop_app(self):
        """Stop all video streams."""
        self.timer.stop()

        for panel in self.camera_panels:
            panel.stop()

        self.update_system_status(False)
        self.set_control_state(False)
        self.update_footer()

    def open_setup(self):
        """Draw and save ROI rectangles for one selected camera source."""
        available_panels = [
            panel
            for panel in self.camera_panels
            if panel.source_path is not None
        ]

        if not available_panels:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select an image/video for ROI setup",
                "",
                "Media Files (*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv *.wmv)",
            )

            if not file_path:
                return

            self.camera_panels[0].load_source(file_path)
            available_panels = [self.camera_panels[0]]

        if len(available_panels) == 1:
            selected_panel = available_panels[0]
        else:
            choices = [
                f"{panel.camera_name}: {Path(panel.source_path).name}"
                for panel in available_panels
            ]

            selected_text, accepted = QInputDialog.getItem(
                self,
                "ROI Setup",
                "Select the camera/source to configure:",
                choices,
                0,
                False,
            )

            if not accepted:
                return

            selected_panel = available_panels[
                choices.index(selected_text)
            ]

        was_running = self.timer.isActive()
        self.timer.stop()

        try:
            ROI_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

            # Prefer the newer ROI helper that can use the frame currently
            # displayed by the app. Fall back to the user's original helper,
            # whose signature only accepts source and config_path.
            try:
                setup_rois(
                    source=Path(selected_panel.source_path),
                    config_path=selected_panel.config_path,
                    reference_frame=selected_panel.current_frame,
                )
            except TypeError as exc:
                if "reference_frame" not in str(exc):
                    raise

                setup_rois(
                    source=Path(selected_panel.source_path),
                    config_path=selected_panel.config_path,
                )

            selected_panel.load_roi_config()

            if selected_panel.current_frame is not None:
                selected_panel.process_and_display(
                    selected_panel.current_frame
                )

            self.collect_stats()

            QMessageBox.information(
                self,
                "ROI Setup",
                "ROI configuration saved successfully.\n\n"
                f"{selected_panel.config_path}",
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "ROI Setup failed",
                f"{type(exc).__name__}: {exc}",
            )

        finally:
            if was_running:
                self.timer.start(30)

    def update_video_frames(self):
        """Update all camera panels."""
        for panel in self.camera_panels:
            try:
                panel.update_frame()
            except Exception as exc:
                self.timer.stop()
                self.update_system_status(False)
                self.set_control_state(False)
                QMessageBox.critical(
                    self,
                    "Detection failed",
                    f"{panel.camera_name}\n\n"
                    f"{type(exc).__name__}: {exc}",
                )
                return

        self.collect_stats()
        self.update_system_status(True)

    def collect_stats(self):
        """Combine the counts of all configured camera panels."""
        counts = [
            panel.get_counts()
            for panel in self.camera_panels
        ]

        self.total_slots = sum(item[0] for item in counts)
        self.free_slots = sum(item[1] for item in counts)
        self.occupied_slots = sum(item[2] for item in counts)

        self.update_stats()

    def set_control_state(self, running):
        """Keep monitoring button text and availability synchronized."""
        if running:
            self.run_button.setText("● MONITORING ACTIVE")
            self.run_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        else:
            self.run_button.setText("START MONITORING")
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def update_system_status(self, running):
        """Show whether video monitoring is active."""
        if running:
            current_time = datetime.now().strftime("%H:%M:%S")

            status_text = (
                "● MONITORING\n"
                f"Last update: {current_time}"
            )
            status_color = "#15803D"
            background_color = "#F0FDF4"
            border_color = "#86EFAC"

        else:
            status_text = (
                "● STOPPED\n"
                "No active monitoring"
            )
            status_color = "#64748B"
            background_color = "#F8FAFC"
            border_color = "#CBD5E1"

        self.status_label.setText(status_text)

        if running != self.monitoring_active:
            self.status_label.setStyleSheet(
                f"""
                QLabel {{
                    color: {status_color};
                    background-color: {background_color};
                    border: 1px solid {border_color};
                    border-radius: 8px;
                    font-size: 13px;
                    font-weight: 700;
                    padding: 8px;
                }}
                """
            )
            self.monitoring_active = running

    def update_footer(self):
        """Show the active model configuration without cluttering the UI."""
        loaded_sources = sum(
            panel.source_path is not None
            for panel in self.camera_panels
        )
        device_name = "CUDA" if torch.cuda.is_available() else "CPU"

        self.footer_label.setText(
            f"Per-camera live analytics"
            f"  ·  {device_name}"
            f"  ·  {loaded_sources} loaded sources"
        )


    def update_stats(self):
        """Refresh the combined parking summary."""
        if self.total_slots > 0:
            occupancy_rate = (
                self.occupied_slots
                / self.total_slots
                * 100
            )
        else:
            occupancy_rate = 0.0

        self.total_label.setText(
            f"TOTAL\n{self.total_slots}"
        )
        self.free_label.setText(
            f"AVAILABLE\n{self.free_slots}"
        )
        self.occupied_label.setText(
            f"OCCUPIED\n{self.occupied_slots}"
        )
        self.rate_label.setText(
            f"OCCUPANCY\n{occupancy_rate:.1f}%"
        )

    def resizeEvent(self, event):
        """Refresh current image scaling after window resize."""
        super().resizeEvent(event)

        for panel in self.camera_panels:
            if panel.last_display_frame is not None:
                panel.display_frame(panel.last_display_frame)

    def closeEvent(self, event):
        self.timer.stop()
        for panel in self.camera_panels:
            panel.stop()
        event.accept()

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #EEF2F6;
                font-family: Segoe UI;
                font-size: 14px;
                color: #172033;
            }

            #TopBar {
                background-color: #FFFFFF;
                border: 1px solid #D8E0EA;
                border-radius: 10px;
            }

            #AppTitle {
                background-color: transparent;
                color: #0F172A;
                font-size: 22px;
                font-weight: 800;
                letter-spacing: 0.5px;
            }

            #AppSubtitle {
                background-color: transparent;
                color: #64748B;
                font-size: 12px;
            }

            #LeftPanel {
                background-color: #0F172A;
                border: 1px solid #1E293B;
                border-radius: 10px;
            }

            #SectionTitle {
                background-color: transparent;
                color: #94A3B8;
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 1px;
                padding: 4px 2px;
            }

            #CameraArea {
                background-color: #E5EAF1;
                border: 1px solid #D8E0EA;
                border-radius: 10px;
            }

            #CameraPanel {
                background-color: #FFFFFF;
                border: 1px solid #D6DEE8;
                border-radius: 9px;
            }

            #CameraHeader {
                background-color: #F8FAFC;
                border: none;
                border-radius: 6px;
            }

            #CameraTitle {
                background-color: transparent;
                color: #1E293B;
                font-size: 12px;
                font-weight: 800;
                letter-spacing: 0.5px;
            }

            #CameraStats {
                background-color: transparent;
                font-size: 12px;
                font-weight: 800;
            }

            #VideoArea {
                background-color: #0B1220;
                color: #94A3B8;
                border-radius: 6px;
                font-size: 15px;
            }

            QPushButton {
                color: #FFFFFF;
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 800;
                padding: 8px;
            }

            #RunButton {
                background-color: #16A34A;
                border: 1px solid #16A34A;
            }

            #RunButton:hover {
                background-color: #15803D;
            }

            #RunButton:disabled {
                background-color: #14532D;
                color: #BBF7D0;
                border: 1px solid #166534;
            }

            #StopButton {
                background-color: transparent;
                color: #F87171;
                border: 1px solid #EF4444;
            }

            #StopButton:hover {
                background-color: #451A1A;
            }

            #StopButton:disabled {
                background-color: transparent;
                color: #475569;
                border: 1px solid #334155;
            }

            #SetupButton {
                background-color: #1E293B;
                color: #E2E8F0;
                border: 1px solid #334155;
            }

            #SetupButton:hover {
                background-color: #334155;
            }

            QPushButton:disabled {
                background-color: #1E293B;
                color: #64748B;
                border: 1px solid #334155;
            }

            #StatBox {
                background-color: #172033;
                color: #E2E8F0;
                border: 1px solid #334155;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 800;
                padding: 6px;
            }

            #StatBox[cardType="available"] {
                background-color: #052E1B;
                color: #86EFAC;
                border: 1px solid #166534;
            }

            #StatBox[cardType="occupied"] {
                background-color: #3F1518;
                color: #FCA5A5;
                border: 1px solid #7F1D1D;
            }

            #StatBox[cardType="rate"] {
                background-color: #172554;
                color: #BFDBFE;
                border: 1px solid #1E40AF;
            }

            #TechnicalFooter {
                background-color: transparent;
                color: #475569;
                font-size: 11px;
                font-weight: 600;
                padding: 0 4px;
            }
        """)


# =========================================================
# 4. MAIN FUNCTION
# =========================================================

def main():
    app = QApplication(sys.argv)

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
