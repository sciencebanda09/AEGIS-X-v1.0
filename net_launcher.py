import time
import logging
from enum import Enum, auto
from typing import Optional

import numpy as np

log = logging.getLogger("NetLauncher")


class LaunchDecision(Enum):
    WAIT   = auto()   # still tracking / aligning
    FIRE   = auto()   # fire now
    ABORT  = auto()   # give up on this target
    RELOAD = auto()   # fired, reloading


class LauncherState(Enum):
    IDLE   = auto()
    COMMIT = auto()
    FIRED  = auto()
    RELOAD = auto()


class NetLauncher:
    """
    Manages launch timing with:
      - Quality threshold gate  (track_quality > commit_tq)
      - Distance window         (min_launch_dist … max_launch_dist)
      - Time-to-go window       (0 < tgo < max_tgo)
      - Stale-guard abort       (> stale_steps without closure)
      - Reload cooldown         (reload_steps after each launch)
    """

    # ── Tuneable parameters ───────────────────────────────────────────────────
    COMMIT_TQ       = 0.55      # track quality needed to commit
    MIN_DIST        = 0.5       # m  — minimum engagement distance
    MAX_DIST        = 4.0       # m  — maximum engagement distance
    MAX_TGO         = 3.0       # s  — abort if tgo larger than this
    COMMIT_STEPS    = 5         # # ticks to hold commit before firing
    STALE_STEPS     = 120       # # ticks before stale abort
    STALE_CLOSURE   = 1.5       # m  — must have closed this much to reset stale
    RELOAD_STEPS    = 25        # # ticks to reload / re-arm (~2 s at 12.5 Hz)

    def __init__(self, arduino_bridge):
        self.arduino         = arduino_bridge
        self._state          = LauncherState.IDLE
        self._commit_count   = 0
        self._stale_count    = 0
        self._reload_count   = 0
        self._prev_dist      = 999.0
        self._total_launches = 0
        self._total_aborts   = 0

    @property
    def state(self) -> LauncherState:
        return self._state

    def decide(
        self,
        target_pos:    np.ndarray,   # 3D [x, y, z]
        track_quality: float,
        tgo:           float,
        dist:          float,
    ) -> LaunchDecision:
        """
        Call every control tick.  Returns LaunchDecision.
        """

        # ── RELOAD cooldown ────────────────────────────────────────────────────
        if self._state == LauncherState.RELOAD:
            self._reload_count += 1
            if self._reload_count >= self.RELOAD_STEPS:
                self._state        = LauncherState.IDLE
                self._reload_count = 0
                log.info("Launcher READY (reload complete)")
            return LaunchDecision.RELOAD

        # ── Quality + window checks ────────────────────────────────────────────
        in_window = (
            track_quality  >= self.COMMIT_TQ and
            self.MIN_DIST  <= dist <= self.MAX_DIST and
            0.0 < tgo      <= self.MAX_TGO
        )

        # ── Stale-guard ────────────────────────────────────────────────────────
        closure = self._prev_dist - dist
        self._prev_dist = dist

        if self._state == LauncherState.COMMIT:
            if closure < 0.01:
                self._stale_count += 1
            else:
                self._stale_count = max(0, self._stale_count - 2)  # reset on closure

            if self._stale_count >= self.STALE_STEPS:
                log.warning(f"STALE ABORT after {self._stale_count} ticks — dist={dist:.2f}m")
                self._state       = LauncherState.IDLE
                self._commit_count = 0
                self._stale_count  = 0
                self._total_aborts += 1
                return LaunchDecision.ABORT

        # ── State machine ──────────────────────────────────────────────────────
        if self._state == LauncherState.IDLE:
            if in_window:
                self._state        = LauncherState.COMMIT
                self._commit_count = 0
                self._stale_count  = 0
                log.info(f"COMMIT  dist={dist:.2f}m  tgo={tgo:.2f}s  tq={track_quality:.2f}")

        elif self._state == LauncherState.COMMIT:
            if not in_window:
                # Target left engagement window
                self._state        = LauncherState.IDLE
                self._commit_count = 0
                return LaunchDecision.WAIT

            self._commit_count += 1

            if self._commit_count >= self.COMMIT_STEPS:
                # ── FIRE ────────────────────────────────────────────────────────
                log.info(
                    f"FIRE  dist={dist:.2f}m  tgo={tgo:.3f}s  "
                    f"tq={track_quality:.2f}  launch#{self._total_launches+1}"
                )
                self._total_launches += 1
                self._state          = LauncherState.RELOAD
                self._commit_count   = 0
                self._stale_count    = 0
                return LaunchDecision.FIRE

        return LaunchDecision.WAIT

    def reset(self):
        """Force back to IDLE (e.g. emergency stop)."""
        self._state        = LauncherState.IDLE
        self._commit_count = 0
        self._stale_count  = 0
        self._reload_count = 0

    def stats(self) -> dict:
        return {
            "launches": self._total_launches,
            "aborts":   self._total_aborts,
            "state":    self._state.name,
        }
