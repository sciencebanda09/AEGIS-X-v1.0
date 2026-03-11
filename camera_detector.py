import math
import time
import threading
import logging
import queue
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger("Camera")

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import cv2
    _CV2_OK = True
except ImportError:
    cv2 = None  # type: ignore
    _CV2_OK = False
    log.warning("OpenCV not found — camera will run in MOCK mode")

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO = None  # type: ignore
    _YOLO_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Detection:
    confidence:    float          # 0–1
    azimuth_deg:   float          # pan angle from camera centre
    elevation_deg: float          # tilt angle from camera centre
    bbox_area:     float          # pixels² — used for depth estimation
    label:         str = "drone"
    frame_x:       float = 0.0   # pixel centre X
    frame_y:       float = 0.0   # pixel centre Y


# ══════════════════════════════════════════════════════════════════════════════
#  YOLO DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
class YOLODetector:
    """YOLOv8n drone detector.  Requires ultralytics package + model file."""

    # COCO classes that could be drones (fallback for generic models)
    DRONE_CLASSES = {"bird", "kite", "airplane", "drone"}

    def __init__(self, model_path: str):
        log.info(f"Loading YOLO model: {model_path}")
        self.model = _YOLO(model_path)
        # Run warmup inference
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        log.info("YOLO model loaded and warmed up")

    def detect(self, frame: np.ndarray) -> List[dict]:
        """Returns list of {x1,y1,x2,y2, conf, label}."""
        results = self.model(frame, verbose=False, conf=0.35)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id   = int(box.cls[0])
                label    = self.model.names[cls_id]
                conf     = float(box.conf[0])
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": conf, "label": label,
                })
        return detections


# ══════════════════════════════════════════════════════════════════════════════
#  BLOB DETECTOR  —  fallback when YOLO not available
# ══════════════════════════════════════════════════════════════════════════════
class BlobDetector:
    """
    Background-subtraction + contour blob detector for moving drones.
    Works well for drones against a clear sky.
    Tuning tips:
      - Increase MIN_AREA if getting false positives on leaves/birds.
      - Adjust HSV range if detecting specific LED colours on drone.
    """

    MIN_AREA  = 200    # px² — minimum contour area
    MAX_AREA  = 8000   # px² — maximum contour area
    MIN_CIRC  = 0.15   # circularity threshold (drones are roughly circular)

    def __init__(self):
        if not _CV2_OK:
            return
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False
        )
        # Optional: simple colour range for orange/red LED drones
        self._hsv_lower = np.array([0,  100, 100], dtype=np.uint8)
        self._hsv_upper = np.array([30, 255, 255], dtype=np.uint8)

    def detect(self, frame: np.ndarray) -> List[dict]:
        if not _CV2_OK:
            return []
        detections = []

        # --- Background subtraction ---
        fg_mask = self._bg.apply(frame)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.MIN_AREA <= area <= self.MAX_AREA):
                continue
            # Circularity check
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue
            circularity = 4 * math.pi * area / (perimeter ** 2)
            if circularity < self.MIN_CIRC:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            conf = min(1.0, circularity * 1.5 * (area / self.MAX_AREA) * 3)
            detections.append({
                "x1": float(x), "y1": float(y),
                "x2": float(x + w), "y2": float(y + h),
                "conf": conf, "label": "drone_blob",
            })

        # --- HSV colour detection (optional — detect bright LEDs) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)
        mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)
        c2, _ = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        for cnt in c2:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append({
                "x1": float(x), "y1": float(y),
                "x2": float(x + w), "y2": float(y + h),
                "conf": 0.6, "label": "drone_led",
            })

        return detections


# ══════════════════════════════════════════════════════════════════════════════
#  PIXEL → ANGLE CONVERSION
# ══════════════════════════════════════════════════════════════════════════════
def bbox_to_angles(
    x1: float, y1: float, x2: float, y2: float,
    frame_w: int, frame_h: int,
    hfov_deg: float = 62.2,   # Pi Camera v2 horizontal FOV
    vfov_deg: float = 48.8,   # Pi Camera v2 vertical FOV
) -> Tuple[float, float, float]:
    """
    Returns (azimuth_deg, elevation_deg, bbox_area_px).
    azimuth  > 0 = right
    elevation> 0 = up
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Normalise to [-0.5, +0.5]
    nx = (cx / frame_w) - 0.5
    ny = 0.5 - (cy / frame_h)     # flip Y so up is positive

    az  = nx * hfov_deg
    el  = ny * vfov_deg
    area = (x2 - x1) * (y2 - y1)

    return float(az), float(el), float(area)


# ══════════════════════════════════════════════════════════════════════════════
#  MOCK CAMERA  —  PC testing without hardware
# ══════════════════════════════════════════════════════════════════════════════
class MockCamera:
    def __init__(self):
        self._t = 0.0

    def read_frame(self):
        self._t += 0.1
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        return True, dummy

    def detections(self) -> List[dict]:
        # Simulate a drone moving across frame
        cx = 320 + 150 * math.sin(self._t * 0.5)
        cy = 240 + 50  * math.sin(self._t * 0.8)
        w, h = 40, 30
        return [{
            "x1": cx - w/2, "y1": cy - h/2,
            "x2": cx + w/2, "y2": cy + h/2,
            "conf": 0.72, "label": "drone_mock",
        }]


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA DETECTOR  —  main class
# ══════════════════════════════════════════════════════════════════════════════
class CameraDetector:
    """
    Background camera thread. Detects drones in each frame,
    converts pixel coordinates to azimuth/elevation angles,
    and pushes Detection objects to the output queue.
    """

    def __init__(
        self,
        camera_index: int  = 0,
        width: int         = 640,
        height: int        = 480,
        fps: int           = 30,
        use_yolo: bool     = False,
        yolo_path: str     = "yolov8n.pt",
        hfov_deg: float    = 62.2,
        vfov_deg: float    = 48.8,
    ):
        self.width     = width
        self.height    = height
        self.hfov_deg  = hfov_deg
        self.vfov_deg  = vfov_deg

        self._use_yolo = use_yolo and _YOLO_OK
        self._mock     = not _CV2_OK

        if self._mock:
            log.warning("Camera using MOCK mode (no OpenCV)")
            self._mock_cam = MockCamera()
            self._detector = None
            self._cap      = None
        else:
            self._cap = cv2.VideoCapture(camera_index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS,          fps)

            if self._use_yolo:
                log.info("Using YOLOv8 detector")
                self._detector = YOLODetector(yolo_path)
            else:
                log.info("Using blob/BG-subtract detector (no YOLO)")
                self._detector = BlobDetector()

        self._thread: Optional[threading.Thread] = None
        self._stop   = threading.Event()

    def start(self, out_queue: queue.Queue):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(out_queue,), daemon=True, name="Camera"
        )
        self._thread.start()
        log.info("Camera thread started")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()

    def _run(self, out_queue: queue.Queue):
        while not self._stop.is_set():
            if self._mock:
                dets_raw = self._mock_cam.detections()
                self._mock_cam.read_frame()  # advance time
            else:
                ret, frame = self._cap.read()
                if not ret:
                    log.warning("Camera read failed")
                    time.sleep(0.05)
                    continue
                dets_raw = self._detector.detect(frame)

            # Convert to Detection objects
            dets = []
            for d in dets_raw:
                az, el, area = bbox_to_angles(
                    d["x1"], d["y1"], d["x2"], d["y2"],
                    self.width, self.height,
                    self.hfov_deg, self.vfov_deg,
                )
                dets.append(Detection(
                    confidence    = float(d["conf"]),
                    azimuth_deg   = az,
                    elevation_deg = el,
                    bbox_area     = area,
                    label         = d.get("label", "drone"),
                    frame_x       = (d["x1"] + d["x2"]) / 2,
                    frame_y       = (d["y1"] + d["y2"]) / 2,
                ))

            if dets:
                try:
                    out_queue.put_nowait(dets)
                except queue.Full:
                    pass

            time.sleep(0.033)  # ~30 fps
