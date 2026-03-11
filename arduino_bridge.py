import time
import threading
import logging
import serial
from typing import Optional

log = logging.getLogger("Arduino")


class ArduinoBridge:
    """Thread-safe serial command bridge to Arduino Uno."""

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0):
        self.port     = port
        self.baudrate = baudrate
        self.timeout  = timeout
        self._ser: Optional[serial.Serial] = None
        self._lock    = threading.Lock()
        self._mock    = False
        self._last_pan  = 90.0
        self._last_tilt = 45.0

    def connect(self):
        try:
            self._ser = serial.Serial(
                port       = self.port,
                baudrate   = self.baudrate,
                timeout    = self.timeout,
                write_timeout = 1.0,
            )
            time.sleep(2.0)   # Wait for Arduino reset after DTR
            self._ser.reset_input_buffer()
            log.info(f"Arduino connected on {self.port} @ {self.baudrate}")
            # Read any startup banner
            for _ in range(10):
                line = self._ser.readline().decode("ascii", errors="ignore").strip()
                if line:
                    log.info(f"Arduino says: {line}")
                else:
                    break
        except serial.SerialException as e:
            log.warning(f"Arduino not found on {self.port}: {e} — using MOCK mode")
            self._mock = True

    def disconnect(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
            log.info("Arduino disconnected")

    def send_command(self, cmd: str) -> bool:
        """Send a raw command string (without newline — added automatically)."""
        if self._mock:
            log.debug(f"[MOCK] → {cmd}")
            return True
        with self._lock:
            try:
                self._ser.write((cmd + "\n").encode("ascii"))
                self._ser.flush()
                return True
            except Exception as e:
                log.error(f"Serial write error: {e}")
                return False

    def send_servo(self, pan_deg: float, tilt_deg: float) -> bool:
        """
        Command launcher servos.
        pan_deg:   0=full left, 90=centre, 180=full right
        tilt_deg:  0=straight down, 90=horizontal, 180=straight up
        Clamped to safe hardware limits.
        """
        pan  = float(max(0.0,  min(180.0, pan_deg)))
        tilt = float(max(20.0, min(130.0, tilt_deg)))  # limit vertical range

        if abs(pan - self._last_pan) < 0.5 and abs(tilt - self._last_tilt) < 0.5:
            return True   # no change — skip transmission

        self._last_pan  = pan
        self._last_tilt = tilt
        return self.send_command(f"SERVO {pan:.1f} {tilt:.1f}")

    def send_drive(self, vx: float, vy: float, omega: float) -> bool:
        """
        Command UGV chassis.
        vx, vy:  forward/lateral velocity -1..+1
        omega:   rotation rate -1..+1
        """
        vx    = max(-1.0, min(1.0, vx))
        vy    = max(-1.0, min(1.0, vy))
        omega = max(-1.0, min(1.0, omega))
        return self.send_command(f"DRIVE {vx:.3f} {vy:.3f} {omega:.3f}")

    def read_feedback(self) -> Optional[str]:
        """Read one line of Arduino feedback (non-blocking)."""
        if self._mock:
            return "OK"
        if not self._ser or not self._ser.in_waiting:
            return None
        try:
            line = self._ser.readline().decode("ascii", errors="ignore").strip()
            return line if line else None
        except Exception:
            return None
