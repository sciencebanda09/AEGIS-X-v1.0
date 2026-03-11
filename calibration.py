import json
import math
import time
import sys
import logging
import serial

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("CALIBRATION")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Find Serial Ports
# ══════════════════════════════════════════════════════════════════════════════
def find_serial_ports():
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    log.info("Available serial ports:")
    for p in ports:
        log.info(f"  {p.device}  — {p.description}  [{p.hwid}]")
    return [p.device for p in ports]


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Servo Calibration
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_servos(arduino_port: str):
    """Sweep servos and ask user to confirm end-stops."""
    log.info(f"Connecting to Arduino on {arduino_port}...")
    try:
        ser = serial.Serial(arduino_port, 115200, timeout=2)
        time.sleep(2.0)
        ser.readline()  # flush banner

        def send(cmd):
            ser.write((cmd + "\n").encode())
            time.sleep(0.05)
            resp = ser.readline().decode("ascii", errors="ignore").strip()
            return resp

        log.info("=== SERVO SWEEP TEST ===")
        log.info("PAN sweep 0 → 180 → 90")
        for pan in range(0, 181, 10):
            resp = send(f"SERVO {pan} 70")
            log.info(f"  PAN={pan}°  resp={resp}")
            time.sleep(0.08)
        send("HOME")

        log.info("TILT sweep 20 → 130 → 70")
        for tilt in range(20, 131, 10):
            resp = send(f"SERVO 90 {tilt}")
            log.info(f"  TILT={tilt}°  resp={resp}")
            time.sleep(0.08)
        send("HOME")

        pan_centre  = input("\nAim pan at a target — enter actual angle offset (+ = right): ")
        tilt_centre = input("Aim tilt at horizon — enter actual angle offset (+ = up):    ")

        cal = {
            "pan_offset":  float(pan_centre),
            "tilt_offset": float(tilt_centre),
        }
        ser.close()
        return cal

    except serial.SerialException as e:
        log.error(f"Cannot open {arduino_port}: {e}")
        return {"pan_offset": 0.0, "tilt_offset": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: Camera Calibration
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_camera(camera_index: int = 0):
    """
    Place a known-size object (e.g. 30cm marker) at a known distance (2m).
    Measure its bounding-box pixel width to calibrate depth estimation.
    """
    try:
        import cv2
    except ImportError:
        log.warning("OpenCV not available — skip camera calibration")
        return {"bbox_ref_area": 4000.0, "ref_depth": 2.0, "hfov_deg": 62.2, "vfov_deg": 48.8}

    log.info("=== CAMERA CALIBRATION ===")
    log.info("Place a 30cm × 30cm marker at exactly 2.0m in front of camera.")
    input("Press Enter when ready...")

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, frame = cap.read()
    if not ret:
        log.error("Camera read failed")
        cap.release()
        return {}

    cv2.imshow("Calibration — select bounding box with mouse, press ENTER", frame)

    bbox = cv2.selectROI("Calibration — select bounding box with mouse, press ENTER",
                         frame, fromCenter=False)
    cv2.destroyAllWindows()
    cap.release()

    if bbox[2] == 0 or bbox[3] == 0:
        log.warning("No ROI selected — using defaults")
        return {"bbox_ref_area": 4000.0, "ref_depth": 2.0}

    area    = float(bbox[2] * bbox[3])
    ref_dep = 2.0   # metres
    log.info(f"Marker bbox area = {area:.0f} px² at {ref_dep}m")
    log.info(f"  → bbox_ref_area = {area:.1f}")

    return {"bbox_ref_area": area, "ref_depth": ref_dep, "hfov_deg": 62.2, "vfov_deg": 48.8}


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: LiDAR Test
# ══════════════════════════════════════════════════════════════════════════════
def test_lidar(lidar_port: str):
    try:
        from lidar_driver import LidarDriver, LidarClusterer
        import queue
        q = queue.Queue()
        driver = LidarDriver(lidar_port)
        driver.start(q)
        log.info("LiDAR running — collecting 5s of data...")
        time.sleep(5.0)
        clusters_seen = 0
        while not q.empty():
            clusters = q.get_nowait()
            clusters_seen += len(clusters)
        driver.stop()
        log.info(f"LiDAR OK — {clusters_seen} clusters detected in 5s")
        return {"lidar_ok": True, "clusters_5s": clusters_seen}
    except Exception as e:
        log.error(f"LiDAR test failed: {e}")
        return {"lidar_ok": False}


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: Net Launcher Test
# ══════════════════════════════════════════════════════════════════════════════
def test_launcher(arduino_port: str):
    confirm = input("\n⚠️  LAUNCH TEST — ensure net launcher is clear! Type YES to continue: ")
    if confirm.strip().upper() != "YES":
        log.info("Launch test skipped")
        return {}
    try:
        ser = serial.Serial(arduino_port, 115200, timeout=2)
        time.sleep(2.0)
        ser.write(b"LAUNCH\n")
        time.sleep(2.0)
        resp = ser.readline().decode("ascii", errors="ignore").strip()
        ser.close()
        log.info(f"Launch test response: {resp}")
        return {"launcher_ok": "DONE" in resp}
    except Exception as e:
        log.error(f"Launcher test failed: {e}")
        return {"launcher_ok": False}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("╔═══════════════════════════════════════╗")
    log.info("║  AEGIS-X CALIBRATION TOOL             ║")
    log.info("╚═══════════════════════════════════════╝")

    ports = find_serial_ports()

    # Interactive port selection
    arduino_port = input(f"\nEnter Arduino port [{'/dev/ttyUSB0'}]: ").strip() or "/dev/ttyUSB0"
    lidar_port   = input(f"Enter LiDAR port  [{'/dev/ttyUSB1'}]: ").strip() or "/dev/ttyUSB1"
    cam_index    = int(input("Camera index [0]: ").strip() or "0")

    results = {
        "arduino_port": arduino_port,
        "lidar_port":   lidar_port,
        "camera_index": cam_index,
    }

    results.update(calibrate_servos(arduino_port))
    results.update(calibrate_camera(cam_index))
    results.update(test_lidar(lidar_port))
    results.update(test_launcher(arduino_port))

    out_path = "calibration.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\n✅  Calibration complete — saved to {out_path}")
    log.info("   Edit main_pi.py CFG dict with these values before running.")
    log.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
