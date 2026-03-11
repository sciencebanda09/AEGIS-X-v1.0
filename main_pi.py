"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        AEGIS-X HARDWARE CONTROLLER  —  Raspberry Pi Main Process           ║
║        Drone Interception System: LiDAR + Camera + Arduino Net Launcher    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Hardware:                                                                  ║
║    • Raspberry Pi 4 (4GB+) — main compute                                  ║
║    • RPLiDAR A1/A2/A3      — 360° 2D range scan (USB/UART)                ║
║    • Pi Camera v2 / USB    — visual detection (OpenCV)                     ║
║    • Arduino Uno           — servo + net launcher PWM control (USB/Serial) ║
║    • UGV chassis           — mecanum/differential drive (via Arduino)      ║
║  Architecture:                                                              ║
║    main_pi.py              — orchestrator, EKF/IMM, assignment             ║
║    lidar_driver.py         — RPLiDAR scan thread                           ║
║    camera_detector.py      — YOLOv8n / colour-blob detection thread        ║
║    arduino_bridge.py       — serial command bridge to Arduino              ║
║    ekf_tracker.py          — IMM-EKF state estimator                       ║
║    guidance.py             — APN intercept geometry                        ║
║    net_launcher.py         — launch timing & commit logic                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import time
import math
import threading
import logging
import queue
import sys
import signal
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import deque

import numpy as np
from numpy.linalg import norm

# ── Local modules ─────────────────────────────────────────────────────────────
from lidar_driver    import LidarDriver, LidarPoint
from camera_detector import CameraDetector, Detection
from arduino_bridge  import ArduinoBridge
from ekf_tracker     import IMMTracker
from guidance        import APNGuidance, compute_intercept_point
from net_launcher    import NetLauncher, LaunchDecision

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/aegis.log"),
    ],
)
log = logging.getLogger("AEGIS-MAIN")

# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM CONFIGURATION  —  edit these for your deployment
# ══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    # Serial ports
    arduino_port    = "/dev/ttyUSB0",   # or /dev/ttyACM0
    lidar_port      = "/dev/ttyUSB1",   # RPLiDAR USB
    arduino_baud    = 115200,
    lidar_baud      = 115200,

    # Camera
    camera_index    = 0,                # 0=PiCamera, 1=USB cam
    camera_width    = 640,
    camera_height   = 480,
    camera_fps      = 30,
    use_yolo        = False,            # True if YOLOv8n model present
    yolo_model_path = "yolov8n.pt",

    # Intercept geometry (metres)
    intercept_r     = 1.0,              # direct-kill radius (net spread)
    frag_r          = 2.5,              # net effective radius
    arena_r         = 8.0,             # max detection radius from UGV
    max_alt         = 5.0,             # max drone height we engage (m)
    min_confidence  = 0.45,            # detection confidence threshold

    # APN guidance
    nav_constant    = 6.5,             # Nc — proportional navigation gain
    max_pan_speed   = 90.0,            # deg/s launcher pan servo
    max_tilt_speed  = 60.0,            # deg/s launcher tilt servo

    # Loop rates
    dt              = 0.08,            # 12.5 Hz main loop
    ekf_dt          = 0.08,

    # Safety
    min_launch_dist = 0.5,             # m — don't fire if drone too close
    max_launch_dist = 4.0,             # m — don't fire if too far
    commit_tq       = 0.55,            # track-quality threshold to commit
    max_tgo         = 3.0,             # s — abort if time-to-go too large
)

# ══════════════════════════════════════════════════════════════════════════════
#  TRACK  — fused drone state estimate
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Track:
    tid: int
    pos: np.ndarray               # [x, y, z] metres (UGV-centred)
    vel: np.ndarray               # [vx, vy, vz] m/s
    cov: np.ndarray               # 3×3 position covariance
    track_quality: float          # 0–1
    age: int = 0
    last_seen: float = 0.0
    source: str = "fusion"        # "lidar", "camera", "fusion"
    ekf: Optional[IMMTracker] = field(default=None, repr=False)

# ══════════════════════════════════════════════════════════════════════════════
#  SENSOR FUSION  —  combines LiDAR clusters + camera detections
# ══════════════════════════════════════════════════════════════════════════════
class SensorFusion:
    """
    Quality-weighted fusion of LiDAR 3D position and camera bearing.
    LiDAR gives accurate range; camera gives bearing + size.
    """

    def __init__(self):
        self._track_counter = 0
        self.tracks: Dict[int, Track] = {}
        self._max_assoc_dist = 1.5   # m — gate for associating measurements

    def _new_tid(self) -> int:
        self._track_counter += 1
        return self._track_counter

    def update(
        self,
        lidar_clusters: List[dict],
        camera_detections: List[Detection],
        t: float,
    ) -> List[Track]:
        """
        Run one fusion cycle. Returns list of active tracks.
        Each lidar_cluster: {pos: [x,y,z], size: float}
        Each Detection: see CameraDetector
        """
        meas_list = []

        # --- LiDAR measurements (high position quality) ---
        for cl in lidar_clusters:
            meas_list.append({
                "pos": np.array(cl["pos"], dtype=float),
                "cov": np.diag([0.05**2, 0.05**2, 0.1**2]),
                "quality": 0.90,
                "source": "lidar",
            })

        # --- Camera bearing measurements (fused with LiDAR depth) ---
        for det in camera_detections:
            if det.confidence < CFG["min_confidence"]:
                continue
            # Camera gives angle, estimate depth from bbox size
            depth_est = self._est_depth_from_bbox(det.bbox_area)
            az  = math.radians(det.azimuth_deg)
            el  = math.radians(det.elevation_deg)
            pos = np.array([
                depth_est * math.cos(el) * math.sin(az),
                depth_est * math.cos(el) * math.cos(az),
                depth_est * math.sin(el),
            ])
            meas_list.append({
                "pos": pos,
                "cov": np.diag([0.3**2, 0.3**2, 0.2**2]),
                "quality": float(det.confidence) * 0.75,
                "source": "camera",
            })

        # --- Associate measurements with existing tracks ---
        unmatched_meas = list(range(len(meas_list)))
        for tid, trk in list(self.tracks.items()):
            best_mi, best_d = None, self._max_assoc_dist
            for mi in unmatched_meas:
                d = float(norm(meas_list[mi]["pos"] - trk.pos))
                if d < best_d:
                    best_d, best_mi = d, mi
            if best_mi is not None:
                m = meas_list[best_mi]
                # Weighted fuse
                w_old = 0.35
                w_new = 0.65
                trk.pos = w_old * trk.pos + w_new * m["pos"]
                trk.cov = w_old * trk.cov + w_new * m["cov"]
                trk.track_quality = min(1.0, trk.track_quality + 0.05)
                trk.age += 1
                trk.last_seen = t
                trk.source = m["source"]
                # EKF update
                if trk.ekf is not None:
                    trk.ekf.update(m["pos"][:2], m["cov"][:2, :2], true_pos=m["pos"][:2])
                    ep = trk.ekf.fpos()
                    trk.pos[:2] = ep
                    ev = trk.ekf.fvel()
                    trk.vel[:2] = ev
                unmatched_meas.remove(best_mi)

        # --- Spawn new tracks for unmatched measurements ---
        for mi in unmatched_meas:
            m = meas_list[mi]
            if m["quality"] > 0.4:
                tid = self._new_tid()
                ekf = IMMTracker(m["pos"][:2].copy(), np.zeros(2))
                self.tracks[tid] = Track(
                    tid=tid,
                    pos=m["pos"].copy(),
                    vel=np.zeros(3),
                    cov=m["cov"],
                    track_quality=m["quality"] * 0.5,
                    age=1,
                    last_seen=t,
                    source=m["source"],
                    ekf=ekf,
                )
                log.info(f"New track T{tid:03d} at {m['pos']} src={m['source']}")

        # --- Prune stale tracks ---
        stale = [tid for tid, trk in self.tracks.items()
                 if (t - trk.last_seen) > 2.0 or trk.track_quality < 0.1]
        for tid in stale:
            log.debug(f"Pruning stale track T{tid:03d}")
            del self.tracks[tid]

        # --- Decay quality of unseen tracks ---
        for trk in self.tracks.values():
            dt = t - trk.last_seen
            if dt > 0.2:
                trk.track_quality = max(0.0, trk.track_quality - 0.02 * dt)

        return list(self.tracks.values())

    @staticmethod
    def _est_depth_from_bbox(bbox_area: float) -> float:
        """
        Estimate depth from bounding-box area using inverse-square law.
        Calibrate BBOX_REF_AREA to your camera at 2m distance.
        """
        BBOX_REF_AREA = 4000.0   # pixels² at 2m — calibrate for your setup
        REF_DEPTH     = 2.0      # metres
        if bbox_area < 1:
            return CFG["max_launch_dist"]
        return REF_DEPTH * math.sqrt(BBOX_REF_AREA / max(bbox_area, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
#  ASSIGNMENT  —  pick best track to engage
# ══════════════════════════════════════════════════════════════════════════════
def select_target(tracks: List[Track]) -> Optional[Track]:
    """
    Lethality-weighted selection: prefer close, high-quality, slow tracks.
    Returns best Track or None.
    """
    if not tracks:
        return None
    best, best_score = None, -1e9
    for trk in tracks:
        dist = float(norm(trk.pos))
        if dist < CFG["min_launch_dist"] or dist > CFG["arena_r"]:
            continue
        if trk.pos[2] > CFG["max_alt"]:
            continue
        # Score: closer + higher quality + lower altitude penalty
        score = (trk.track_quality * 2.0
                 - dist / CFG["arena_r"]
                 - abs(trk.pos[2]) * 0.1)
        if score > best_score:
            best_score, best = score, trk
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AEGIS CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════
class AEGISController:
    """
    Main orchestrator — runs sense → track → assign → engage pipeline.
    """

    def __init__(self):
        log.info("=== AEGIS-X HARDWARE CONTROLLER STARTING ===")

        # Hardware interfaces
        self.arduino  = ArduinoBridge(CFG["arduino_port"], CFG["arduino_baud"])
        self.lidar    = LidarDriver(CFG["lidar_port"],    CFG["lidar_baud"])
        self.camera   = CameraDetector(
            camera_index = CFG["camera_index"],
            width        = CFG["camera_width"],
            height       = CFG["camera_height"],
            fps          = CFG["camera_fps"],
            use_yolo     = CFG["use_yolo"],
            yolo_path    = CFG["yolo_model_path"],
        )

        # Processing modules
        self.fusion   = SensorFusion()
        self.guidance = APNGuidance(nav_constant=CFG["nav_constant"])
        self.launcher = NetLauncher(self.arduino)

        # State
        self.running        = False
        self.t              = 0.0
        self.step_count     = 0
        self.active_target: Optional[Track] = None
        self.kills          = 0
        self.misses         = 0
        self.latencies      = deque(maxlen=200)

        # Thread-safe queues for sensor data
        self._lidar_q  = queue.Queue(maxsize=5)
        self._camera_q = queue.Queue(maxsize=5)

        # Control flags
        self._stop_event = threading.Event()

    # ── Startup / shutdown ──────────────────────────────────────────────────

    def start(self):
        log.info("Connecting hardware...")
        self.arduino.connect()
        self.lidar.start(self._lidar_q)
        self.camera.start(self._camera_q)

        # Home the launcher
        self.arduino.send_command("HOME")
        time.sleep(1.5)

        self.running = True
        log.info("All hardware online — entering main loop")
        self._run_loop()

    def stop(self):
        log.info("Shutting down AEGIS...")
        self._stop_event.set()
        self.running = False
        self.lidar.stop()
        self.camera.stop()
        self.arduino.send_command("SAFE")
        self.arduino.disconnect()
        log.info(f"Session summary: kills={self.kills} misses={self.misses}")

    # ── Main control loop ────────────────────────────────────────────────────

    def _run_loop(self):
        dt = CFG["dt"]
        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            # 1. Collect latest sensor data (non-blocking)
            lidar_clusters = self._drain_lidar()
            camera_dets    = self._drain_camera()

            # 2. Fuse & track
            tracks = self.fusion.update(lidar_clusters, camera_dets, self.t)

            # 3. Target selection
            target = select_target(tracks)

            # 4. Guidance & launch decision
            if target is not None:
                self._engage(target)
            else:
                # No target — pan to scan position
                self.arduino.send_command("SCAN")
                self.active_target = None

            # 5. Timing
            elapsed = time.perf_counter() - t0
            self.latencies.append(elapsed * 1000)  # ms
            self.t          += dt
            self.step_count += 1

            sleep_t = max(0.0, dt - elapsed)
            time.sleep(sleep_t)

            # 6. Periodic status print
            if self.step_count % 50 == 0:
                n_tracks = len(tracks)
                med_lat  = float(np.median(self.latencies)) if self.latencies else 0
                log.info(
                    f"t={self.t:.1f}s  tracks={n_tracks}  "
                    f"kills={self.kills}  lat={med_lat:.1f}ms"
                )

    # ── Sensing ─────────────────────────────────────────────────────────────

    def _drain_lidar(self) -> List[dict]:
        clusters = []
        try:
            while True:
                item = self._lidar_q.get_nowait()
                clusters.extend(item)
        except queue.Empty:
            pass
        return clusters

    def _drain_camera(self) -> List[Detection]:
        dets = []
        try:
            while True:
                item = self._camera_q.get_nowait()
                dets.extend(item)
        except queue.Empty:
            pass
        return dets

    # ── Engagement ──────────────────────────────────────────────────────────

    def _engage(self, target: Track):
        """
        Compute APN guidance angles, command servos, decide launch.
        """
        tpos = target.pos           # [x, y, z]
        tvel = target.vel           # [vx, vy, vz]
        dist = float(norm(tpos))

        # --- Compute intercept point (lead-angle aim) ---
        intercept_pt, tgo = compute_intercept_point(
            target_pos  = tpos,
            target_vel  = tvel,
            launcher_pos= np.zeros(3),
            net_speed   = 8.0,           # m/s — net throw speed, calibrate
        )

        # --- APN azimuth / elevation command ---
        pan_deg, tilt_deg = self.guidance.compute_angles(intercept_pt)

        # --- Send servo angles to Arduino ---
        self.arduino.send_servo(pan_deg, tilt_deg)
        self.active_target = target

        # --- Launch decision ---
        decision = self.launcher.decide(
            target_pos   = tpos,
            track_quality= target.track_quality,
            tgo          = tgo,
            dist         = dist,
        )

        if decision == LaunchDecision.FIRE:
            log.info(
                f"🚀 LAUNCH  T{target.tid:03d}  "
                f"dist={dist:.2f}m  tgo={tgo:.2f}s  "
                f"pan={pan_deg:.1f}°  tilt={tilt_deg:.1f}°"
            )
            self.arduino.send_command("LAUNCH")
            self.kills += 1
            # Re-home after launch
            time.sleep(0.3)
            self.arduino.send_command("HOME")

        elif decision == LaunchDecision.ABORT:
            log.warning(f"ABORT engagement T{target.tid:03d}")
            self.misses += 1
            self.active_target = None


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    controller = AEGISController()

    def _sigint(sig, frame):
        log.info("SIGINT received — stopping")
        controller.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    try:
        controller.start()
    except Exception as e:
        log.exception(f"Fatal error: {e}")
        controller.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
