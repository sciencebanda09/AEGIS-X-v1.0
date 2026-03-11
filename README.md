# AEGIS-X Hardware Controller — Requirements & Setup Guide

## Python packages (Raspberry Pi)

```
# Core
numpy>=1.24.0
scipy>=1.10.0
pyserial>=3.5

# LiDAR (choose one based on your RPLiDAR library)
rplidar-roboticia>=0.9.2
# OR: pyrplidar>=0.1.2

# Camera
opencv-python>=4.8.0   # or opencv-python-headless on Pi

# Optional: YOLOv8 detection (requires ~500MB, install only if Pi has 4GB RAM)
# ultralytics>=8.0.0
```

Install with:
```bash
pip install numpy scipy pyserial rplidar-roboticia opencv-python
```

For YOLOv8 (optional):
```bash
pip install ultralytics
```

## Hardware Wiring Summary

### Raspberry Pi → Arduino (USB Cable)
- Plug Arduino USB into any Raspberry Pi USB port
- It appears as `/dev/ttyUSB0` or `/dev/ttyACM0`

### Arduino Pin Connections
| Arduino Pin | Connected To         | Notes                          |
|-------------|----------------------|--------------------------------|
| D3          | Pan servo signal     | 5V servo, orange/yellow wire   |
| D5          | Tilt servo signal    | 5V servo                       |
| D6          | Launch servo/relay   | Net throw trigger              |
| D7          | LED red cathode      | Via 220Ω resistor              |
| D8          | LED green cathode    | Via 220Ω resistor              |
| D9–D12      | L298N motor driver   | For UGV chassis                |
| A0          | Battery voltage divider | 10kΩ/3.3kΩ to 12V battery  |
| 5V          | Servo VCC            | Use external 5V BEC for servos |
| GND         | Common ground        |                                |

### RPLiDAR → Raspberry Pi
| RPLiDAR | Raspberry Pi         |
|---------|----------------------|
| USB     | USB port → /dev/ttyUSB1 |
| VCC     | 5V                   |
| GND     | GND                  |

### Pi Camera v2
- Connect ribbon cable to CSI camera port (not USB)
- Enable: `sudo raspi-config` → Interface Options → Camera

## Setup Steps

1. **Flash Arduino firmware**
   ```
   arduino_firmware/aegis_arduino/aegis_arduino.ino
   ```
   Open in Arduino IDE, select "Arduino Uno", upload.

2. **Install Python dependencies**
   ```bash
   pip install numpy scipy pyserial rplidar-roboticia opencv-python
   ```

3. **Enable Pi Camera** (if using ribbon camera)
   ```bash
   sudo raspi-config
   # Interface Options → Legacy Camera → Enable
   # Reboot
   ```

4. **Run calibration**
   ```bash
   python calibration.py
   ```
   Follow prompts to verify all hardware and save `calibration.json`.

5. **Update CFG in main_pi.py**
   Edit the `CFG` dict at the top of `main_pi.py` with your port names
   and calibrated parameters from `calibration.json`.

6. **Run the system**
   ```bash
   python main_pi.py
   ```

## Tuning Parameters (in main_pi.py CFG)

| Parameter         | Default | Description                                    |
|-------------------|---------|------------------------------------------------|
| `intercept_r`     | 1.0m    | Direct kill radius (net spread diameter/2)     |
| `frag_r`          | 2.5m    | Net effective capture radius                   |
| `arena_r`         | 8.0m    | Max detection/engagement range                 |
| `max_alt`         | 5.0m    | Max drone altitude to engage                   |
| `nav_constant`    | 6.5     | APN Nc — higher = more aggressive guidance     |
| `commit_tq`       | 0.55    | Track quality threshold to commit to launch    |
| `max_tgo`         | 3.0s    | Abort if time-to-intercept > this              |
| `min_launch_dist` | 0.5m    | Safety: don't fire if drone this close         |
| `max_launch_dist` | 4.0m    | Don't fire if drone this far                   |

## Architecture Diagram

```
RPLiDAR A1/A2  →  lidar_driver.py  ─┐
Pi Camera v2   →  camera_detector.py ┼→  main_pi.py  →  arduino_bridge.py  →  Arduino Uno
                                     │   (fusion +        (serial cmd)          (servos +
                                     │    EKF +                                  launcher +
                                     │    APN +                                  motors)
                                     │    assignment)
                                     └── ekf_tracker.py
                                         guidance.py
                                         net_launcher.py
```
