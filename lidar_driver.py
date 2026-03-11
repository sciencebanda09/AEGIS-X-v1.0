import math
import time
import threading
import logging
import queue
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

log = logging.getLogger("LiDAR")

# ── Attempt to import RPLiDAR library ─────────────────────────────────────────
try:
    from rplidar import RPLidar as _RPLidar
    _LIDAR_LIB = "rplidar"
except ImportError:
    try:
        from pyrplidar import PyRPlidar as _RPLidar   # type: ignore
        _LIDAR_LIB = "pyrplidar"
    except ImportError:
        _RPLidar = None
        _LIDAR_LIB = "none"
        log.warning("No RPLiDAR library found — running in MOCK mode")


@dataclass
class LidarPoint:
    angle_deg: float    # 0–360, 0=forward
    distance_m: float  # metres
    quality: int        # 0–15


# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTERING  —  groups nearby points into drone candidates
# ══════════════════════════════════════════════════════════════════════════════
class LidarClusterer:
    """
    Simple angular-gap clustering on a 2D LiDAR scan.
    Converts polar clusters to 3D XYZ assuming drone is at estimated altitude.
    """

    # Tunable parameters
    MIN_POINTS      = 3      # minimum scan points to form a cluster
    GAP_THRESHOLD   = 0.4    # m  — range gap that splits clusters
    MIN_RANGE       = 0.30   # m  — ignore returns closer than this
    MAX_RANGE       = 8.0    # m  — ignore returns farther than this
    MIN_CLUSTER_R   = 0.05   # m  — cluster diameter min
    MAX_CLUSTER_R   = 0.60   # m  — cluster diameter max (drone body)
    DRONE_ALT_EST   = 1.5    # m  — assumed drone altitude above LiDAR

    def cluster(self, scan_points: List[LidarPoint]) -> List[dict]:
        """
        Input:  list of LidarPoint
        Output: list of {pos:[x,y,z], size:float, n_points:int}
        """
        # Filter
        pts = [p for p in scan_points
               if self.MIN_RANGE < p.distance_m < self.MAX_RANGE and p.quality > 2]
        if len(pts) < self.MIN_POINTS:
            return []

        # Sort by angle
        pts.sort(key=lambda p: p.angle_deg)

        # Angular + range gap clustering
        clusters = []
        cur = [pts[0]]
        for prev, curr in zip(pts, pts[1:]):
            ang_gap_m = abs(prev.distance_m - curr.distance_m)
            ang_diff  = abs(curr.angle_deg - prev.angle_deg)
            if ang_diff > 180:
                ang_diff = 360 - ang_diff
            gap_ok = ang_gap_m < self.GAP_THRESHOLD and ang_diff < 15.0
            if gap_ok:
                cur.append(curr)
            else:
                if len(cur) >= self.MIN_POINTS:
                    clusters.append(cur)
                cur = [curr]
        if len(cur) >= self.MIN_POINTS:
            clusters.append(cur)

        # Convert clusters to 3D positions
        results = []
        for cl in clusters:
            # Average polar position
            avg_dist  = np.mean([p.distance_m for p in cl])
            angles_r  = [math.radians(p.angle_deg) for p in cl]
            avg_angle = math.atan2(
                np.mean([math.sin(a) for a in angles_r]),
                np.mean([math.cos(a) for a in angles_r])
            )
            # Cluster diameter (spread check)
            dists = [p.distance_m for p in cl]
            # Compute angular span as chord length
            span = max(dists) - min(dists)
            if not (self.MIN_CLUSTER_R <= span <= self.MAX_CLUSTER_R):
                # Also check angular width * avg_dist
                ang_span_r = max(angles_r) - min(angles_r)
                chord = avg_dist * ang_span_r
                if not (self.MIN_CLUSTER_R <= chord <= self.MAX_CLUSTER_R * 3):
                    continue

            # 2D position (LiDAR plane = horizontal at sensor height)
            # X = right, Y = forward, Z = up (estimated)
            x = avg_dist * math.sin(avg_angle)   # lateral
            y = avg_dist * math.cos(avg_angle)   # forward
            z = self.DRONE_ALT_EST               # estimated altitude

            results.append({
                "pos": [x, y, z],
                "size": float(max(dists) - min(dists)),
                "n_points": len(cl),
                "distance": avg_dist,
            })

        return results


# ══════════════════════════════════════════════════════════════════════════════
#  MOCK LiDAR  —  used when hardware not present (testing on PC)
# ══════════════════════════════════════════════════════════════════════════════
class MockLidar:
    """Generates fake drone detections for testing without hardware."""

    def __init__(self):
        self._t = 0.0
        self._drone_angle = 45.0
        self._drone_dist  = 3.0

    def scan(self):
        """Yields one full scan worth of LidarPoints."""
        import random
        self._t += 0.1
        self._drone_angle = (self._drone_angle + 3.0) % 360
        self._drone_dist  = 3.0 + 1.5 * math.sin(self._t * 0.4)

        pts = []
        for ang in range(0, 360, 2):
            dist = 7.5 + random.gauss(0, 0.05)   # background wall
            q    = 10
            # Inject fake drone cluster around drone_angle
            diff = abs(ang - self._drone_angle)
            if diff > 180:
                diff = 360 - diff
            if diff < 4:
                dist = self._drone_dist + random.gauss(0, 0.04)
                q    = 15
            pts.append(LidarPoint(float(ang), dist, q))
        return pts


# ══════════════════════════════════════════════════════════════════════════════
#  LIDAR DRIVER
# ══════════════════════════════════════════════════════════════════════════════
class LidarDriver:
    """
    Background thread that reads RPLiDAR scans, clusters them,
    and pushes drone candidates to an output queue.
    """

    def __init__(self, port: str, baudrate: int = 115200):
        self.port     = port
        self.baudrate = baudrate
        self._thread: Optional[threading.Thread] = None
        self._stop    = threading.Event()
        self._clusterer = LidarClusterer()
        self._mock      = (_LIDAR_LIB == "none")
        if self._mock:
            log.warning("LiDAR using MOCK mode — no real hardware")
            self._mock_hw = MockLidar()

    def start(self, out_queue: queue.Queue):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(out_queue,), daemon=True, name="LiDAR"
        )
        self._thread.start()
        log.info(f"LiDAR thread started (port={self.port}, lib={_LIDAR_LIB})")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self, out_queue: queue.Queue):
        if self._mock:
            self._run_mock(out_queue)
        else:
            self._run_real(out_queue)

    def _run_mock(self, out_queue: queue.Queue):
        while not self._stop.is_set():
            scan = self._mock_hw.scan()
            clusters = self._clusterer.cluster(scan)
            if clusters:
                try:
                    out_queue.put_nowait(clusters)
                except queue.Full:
                    pass
            time.sleep(0.1)

    def _run_real(self, out_queue: queue.Queue):
        try:
            if _LIDAR_LIB == "rplidar":
                lidar = _RPLidar(self.port, self.baudrate, timeout=3)
                lidar.start_motor()
                time.sleep(1.0)
                for scan in lidar.iter_scans(scan_type='normal', min_len=5):
                    if self._stop.is_set():
                        break
                    pts = [LidarPoint(float(a), float(d) / 1000.0, int(q))
                           for q, a, d in scan]
                    clusters = self._clusterer.cluster(pts)
                    if clusters:
                        try:
                            out_queue.put_nowait(clusters)
                        except queue.Full:
                            pass
                lidar.stop()
                lidar.stop_motor()
                lidar.disconnect()

            elif _LIDAR_LIB == "pyrplidar":
                lidar = _RPLidar()
                lidar.connect(port=self.port, baudrate=self.baudrate, timeout=3)
                lidar.set_motor_pwm(660)
                time.sleep(2)
                for scan in lidar.force_scan():
                    if self._stop.is_set():
                        break
                    pts = [LidarPoint(float(m.angle), float(m.distance) / 1000.0, 15)
                           for m in scan]
                    clusters = self._clusterer.cluster(pts)
                    if clusters:
                        try:
                            out_queue.put_nowait(clusters)
                        except queue.Full:
                            pass
                lidar.stop()
                lidar.disconnect()

        except Exception as e:
            log.error(f"LiDAR error: {e}")
