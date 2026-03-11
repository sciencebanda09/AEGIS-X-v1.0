import math
from typing import Tuple

import numpy as np
from numpy.linalg import norm


# ══════════════════════════════════════════════════════════════════════════════
#  INTERCEPT POINT PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def compute_intercept_point(
    target_pos:   np.ndarray,   # [x, y, z]  metres
    target_vel:   np.ndarray,   # [vx, vy, vz] m/s
    launcher_pos: np.ndarray,   # [x, y, z]  (usually [0,0,0])
    net_speed:    float = 8.0,  # m/s — net / projectile launch speed
    max_iter:     int   = 10,
) -> Tuple[np.ndarray, float]:
    """
    Iterative intercept point for a constant-velocity drone.
    Returns (intercept_pos_3d, time_to_go_s).

    Algorithm:
      t_go ← dist / net_speed
      Loop: predict target at t_go, compute new t_go from new dist
    """
    pos = np.array(target_pos, dtype=float)
    vel = np.array(target_vel, dtype=float)
    lpos = np.array(launcher_pos, dtype=float)

    dist = float(norm(pos - lpos))
    tgo  = dist / (net_speed + 1e-6)

    for _ in range(max_iter):
        pred_pos = pos + vel * tgo
        new_dist = float(norm(pred_pos - lpos))
        tgo_new  = new_dist / (net_speed + 1e-6)
        if abs(tgo_new - tgo) < 1e-4:
            tgo = tgo_new
            break
        tgo = tgo_new

    intercept_pt = pos + vel * tgo
    return intercept_pt, float(tgo)


# ══════════════════════════════════════════════════════════════════════════════
#  APN GUIDANCE
# ══════════════════════════════════════════════════════════════════════════════
class APNGuidance:
    
    def __init__(self, nav_constant: float = 6.5):
        self.Nc = nav_constant
        self._prev_vel_target = np.zeros(3)
        self._dt = 0.08

    def compute_angles(self, aim_point: np.ndarray) -> Tuple[float, float]:
        """
        Convert 3D aim point [x, y, z] (launcher-frame) to
        (pan_deg, tilt_deg) servo angles.

        Launcher frame:
          +Y = forward
          +X = right
          +Z = up

        Pan  = 90° at centre; 0° = full left; 180° = full right
        Tilt = 90° = horizontal; >90° = tilted up
        """
        x, y, z = float(aim_point[0]), float(aim_point[1]), float(aim_point[2])

        # Azimuth (pan): angle in XY horizontal plane, 0 = forward
        azimuth_rad = math.atan2(x, max(y, 0.01))   # limit forward div-by-zero

        # Elevation (tilt): angle above horizontal
        horiz_dist  = math.sqrt(x**2 + y**2)
        elevation_rad = math.atan2(z, max(horiz_dist, 0.01))

        # Convert to servo degrees
        pan_deg  = 90.0 + math.degrees(azimuth_rad)     # 90 = centre
        tilt_deg = 90.0 + math.degrees(elevation_rad)   # 90 = horizontal

        # Clamp to physical limits
        pan_deg  = max(0.0,  min(180.0, pan_deg))
        tilt_deg = max(20.0, min(150.0, tilt_deg))

        return float(pan_deg), float(tilt_deg)

    def compute_apn_aim(
        self,
        target_pos: np.ndarray,   # 3D [x, y, z]
        target_vel: np.ndarray,   # 3D [vx, vy, vz]
        launcher_pos: np.ndarray, # 3D (usually zeros)
        launcher_vel: np.ndarray, # 3D launcher velocity (usually zeros for static)
        dt: float = 0.08,
    ) -> np.ndarray:
        """
        Full APN computation returning the commanded aim direction.
        Returns normalised aim vector [3D].
        """
        rel_pos = target_pos - launcher_pos
        rel_vel = target_vel - launcher_vel
        r       = float(norm(rel_pos))

        if r < 0.01:
            return np.array([0.0, 1.0, 0.0])   # straight ahead fallback

        r_hat  = rel_pos / r
        Vc     = -float(np.dot(rel_vel, r_hat))   # closing rate

        # Perp to LOS (2D in XY, extended to 3D)
        e_perp = np.array([-r_hat[1], r_hat[0], 0.0])
        lam_dot = float(np.dot(rel_vel, e_perp)) / (r + 1e-4)

        # Target acceleration estimate (first difference of velocity)
        a_target = (target_vel - self._prev_vel_target) / (dt + 1e-8)
        self._prev_vel_target = target_vel.copy()

        # APN command acceleration magnitude
        a_pn  = self.Nc * Vc * lam_dot
        a_ff  = self.Nc * float(np.dot(a_target, e_perp))

        a_cmd_mag = a_pn + a_ff

        # Aim direction: current LOS + perpendicular correction
        aim_dir = r_hat + (e_perp * a_cmd_mag * dt / max(Vc, 0.1))
        aim_norm = float(norm(aim_dir))
        if aim_norm > 1e-6:
            aim_dir = aim_dir / aim_norm

        return aim_dir

    def lead_angle_aim(
        self,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        net_speed: float = 8.0,
        lookahead_bias: float = 1.4,
    ) -> np.ndarray:
        """
        Simple lead-angle aim point using travel-time prediction.
        Returns 3D aim point.
        """
        dist = float(norm(target_pos))
        t_go = dist / (net_speed + 1e-6) * lookahead_bias
        return target_pos + target_vel * t_go


# ══════════════════════════════════════════════════════════════════════════════
#  ZERO-EFFORT MISS (diagnostic helper)
# ══════════════════════════════════════════════════════════════════════════════
def zero_effort_miss(
    r: float,
    lam_dot: float,
    tgo: float,
    Nc: float = 6.5,
    a_target: float = 0.0,
) -> float:
    """
    ZEM_APN = r * lam_dot * tgo / (Nc - 1)  -  a_target * tgo² / (2*Nc)
    """
    if Nc <= 1:
        return 1e6
    zem_pn  = r * lam_dot * tgo / (Nc - 1)
    zem_ff  = a_target * tgo ** 2 / (2 * Nc)
    return abs(zem_pn - zem_ff)
