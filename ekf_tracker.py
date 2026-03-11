import math
from collections import deque
from typing import List, Optional

import numpy as np
from numpy.linalg import inv, slogdet

DT_DEFAULT = 0.08   # seconds — overridden per call


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE EKF MODEL
# ══════════════════════════════════════════════════════════════════════════════
class KalmanModel:
    """One EKF model (CV / CA / CT / Singer)."""

    def __init__(self, name: str, n: int, q_scale: float):
        self.name    = name
        self.n       = n
        self.q_scale = q_scale
        self.x = np.zeros(n)      # state
        self.P = np.eye(n) * 50.0 # covariance

    def pos(self) -> np.ndarray:
        return self.x[:2].copy()

    def vel(self) -> np.ndarray:
        return self.x[2:4].copy()

    def _make_F(self, dt: float) -> np.ndarray:
        raise NotImplementedError

    def _make_Q(self, dt: float) -> np.ndarray:
        return np.eye(self.n) * self.q_scale

    def predict(self, dt: float):
        F = self._make_F(dt)
        Q = self._make_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, R: np.ndarray) -> float:
        """Returns Gaussian log-likelihood."""
        H = np.zeros((2, self.n)); H[0, 0] = 1.0; H[1, 1] = 1.0
        innov = z - H @ self.x
        S     = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ inv(S)
        except np.linalg.LinAlgError:
            return 1e-300
        self.x  = self.x + K @ innov
        self.P  = (np.eye(self.n) - K @ H) @ self.P
        # Log-likelihood
        sign, ld = slogdet(S)
        if sign <= 0:
            return 1e-300
        ll = -0.5 * (innov @ inv(S) @ innov + ld + 2 * math.log(2 * math.pi))
        return float(max(math.exp(ll), 1e-300))

    def predict_pos(self, steps: int, dt: float) -> np.ndarray:
        F = self._make_F(dt)
        Fn = np.linalg.matrix_power(F, steps)
        return (Fn @ self.x)[:2]


# ── CV ────────────────────────────────────────────────────────────────────────
class CVModel(KalmanModel):
    def __init__(self):
        super().__init__("CV", n=4, q_scale=0.20)

    def _make_F(self, dt: float) -> np.ndarray:
        return np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ])


# ── CA ────────────────────────────────────────────────────────────────────────
class CAModel(KalmanModel):
    def __init__(self):
        super().__init__("CA", n=6, q_scale=1.00)

    def _make_F(self, dt: float) -> np.ndarray:
        dt2 = 0.5 * dt * dt
        return np.array([
            [1, 0, dt,  0, dt2,   0],
            [0, 1,  0, dt,   0, dt2],
            [0, 0,  1,  0,  dt,   0],
            [0, 0,  0,  1,   0,  dt],
            [0, 0,  0,  0,   1,   0],
            [0, 0,  0,  0,   0,   1],
        ])


# ── CT ────────────────────────────────────────────────────────────────────────
class CTModel(KalmanModel):
    def __init__(self):
        super().__init__("CT", n=5, q_scale=2.20)
        self._omega = 0.15   # rad/s initial turn rate

    def _make_F(self, dt: float) -> np.ndarray:
        w = self._omega
        if abs(w) < 1e-4:
            # Degenerate to CV
            return np.array([
                [1, 0, dt,  0, 0],
                [0, 1,  0, dt, 0],
                [0, 0,  1,  0, 0],
                [0, 0,  0,  1, 0],
                [0, 0,  0,  0, 1],
            ])
        sw = math.sin(w * dt) / w
        cw = (1 - math.cos(w * dt)) / w
        return np.array([
            [1, 0,  sw, -cw, 0],
            [0, 1,  cw,  sw, 0],
            [0, 0,  math.cos(w*dt), -math.sin(w*dt), 0],
            [0, 0,  math.sin(w*dt),  math.cos(w*dt), 0],
            [0, 0,  0,   0, 1],
        ])

    def update(self, z: np.ndarray, R: np.ndarray) -> float:
        ll = super().update(z, R)
        self._omega = float(np.clip(self.x[4], -0.6, 0.6))
        return ll


# ── Singer ────────────────────────────────────────────────────────────────────
class SingerModel(KalmanModel):
    ALPHA = 0.5   # manoeuvre decay constant s⁻¹

    def __init__(self):
        super().__init__("SG", n=6, q_scale=2.80)

    def _make_F(self, dt: float) -> np.ndarray:
        a = self.ALPHA
        e = math.exp(-a * dt)
        a2 = a * a
        return np.array([
            [1, 0, dt,  0, (e + a*dt - 1)/a2,              0],
            [0, 1,  0, dt,               0, (e + a*dt - 1)/a2],
            [0, 0,  1,  0,       (1 - e)/a,               0],
            [0, 0,  0,  1,               0,       (1 - e)/a],
            [0, 0,  0,  0,               e,               0],
            [0, 0,  0,  0,               0,               e],
        ])


# ══════════════════════════════════════════════════════════════════════════════
#  IMM TRACKER
# ══════════════════════════════════════════════════════════════════════════════
class IMMTracker:
    """
    Adaptive Interacting Multiple Model EKF tracker.
    State: [x, y, vx, vy] fused from four motion models.
    """

    # Base Markov transition matrix  (rows=from, cols=to)
    PI_BASE = np.array([
        [0.75, 0.10, 0.10, 0.05],
        [0.10, 0.70, 0.12, 0.08],
        [0.10, 0.12, 0.70, 0.08],
        [0.05, 0.10, 0.10, 0.75],
    ])

    def __init__(self, init_pos: np.ndarray, init_vel: np.ndarray):
        self.models: List[KalmanModel] = [CVModel(), CAModel(), CTModel(), SingerModel()]
        N = len(self.models)

        # Initialise each model
        for m in self.models:
            m.x[:2] = init_pos
            if m.n >= 4:
                m.x[2:4] = init_vel

        self.mu = np.array([0.40, 0.25, 0.20, 0.15])   # model probabilities
        self.Pi = self.PI_BASE.copy()

        self._mode_hist = deque(maxlen=100)
        self._innov     = deque(maxlen=60)
        self._step_count = 0

        # History for diagnostics
        self.est_hist  = deque(maxlen=200)
        self.meas_hist = deque(maxlen=200)
        self.true_hist = deque(maxlen=200)

    # ── One IMM step ──────────────────────────────────────────────────────────

    def update(
        self,
        z: np.ndarray,
        R: np.ndarray,
        dt: float = DT_DEFAULT,
        true_pos: Optional[np.ndarray] = None,
    ):
        """
        z  : 2D position measurement [x, y]
        R  : 2×2 measurement noise covariance
        dt : timestep in seconds
        """
        self._step_count += 1
        N = len(self.models)
        mu = self.mu

        # ── Step 1: Predicted model probs ────────────────────────────────────
        mu_bar = self.Pi.T @ mu   # shape (N,)

        # ── Step 2: Mixing weights ────────────────────────────────────────────
        M = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                M[i, j] = self.Pi[i, j] * mu[i] / (mu_bar[j] + 1e-12)

        # ── Step 3: Mixed initial conditions ──────────────────────────────────
        x_mix = []
        P_mix = []
        for j in range(N):
            nj = self.models[j].n
            xj = np.zeros(nj)
            for i in range(N):
                ni = self.models[i].n
                xi = self.models[i].x
                xj += M[i, j] * (xi[:nj] if ni >= nj
                                  else np.pad(xi, (0, nj - ni)))
            Pj = np.zeros((nj, nj))
            for i in range(N):
                ni = self.models[i].n
                xi = self.models[i].x
                dx = (xi[:nj] if ni >= nj else np.pad(xi, (0, nj - ni))) - xj
                Pi_ext = np.zeros((nj, nj))
                s = min(ni, nj)
                Pi_ext[:s, :s] = self.models[i].P[:s, :s]
                Pj += M[i, j] * (Pi_ext + np.outer(dx, dx))
            x_mix.append(xj)
            P_mix.append(Pj)

        # ── Step 4: Mode-conditioned EKF predict + update ─────────────────────
        likelihoods = []
        for j, m in enumerate(self.models):
            m.x[:] = x_mix[j]
            m.P[:] = P_mix[j]
            m.predict(dt)
            lik = m.update(z, R)
            likelihoods.append(lik)

        # ── Step 5: Update model probabilities ────────────────────────────────
        c  = np.array(likelihoods) * mu_bar
        cs = c.sum()
        self.mu = np.clip(c / (cs if cs > 0 else 1.0), 1e-4, 1.0)
        self.mu /= self.mu.sum()

        # ── Step 6: Adaptive transition matrix (every 50 steps) ───────────────
        dom = int(np.argmax(self.mu))
        self._mode_hist.append(dom)
        if self._step_count % 50 == 0 and len(self._mode_hist) >= 10:
            self._adapt_transitions()

        # Track innovation
        innov = z - self.fpos()
        self._innov.append(float(np.linalg.norm(innov)))

        self.est_hist.append(self.fpos().copy())
        self.meas_hist.append(z.copy())
        if true_pos is not None:
            self.true_hist.append(true_pos.copy())

    def _adapt_transitions(self):
        N = len(self.models)
        hist = list(self._mode_hist)
        counts = np.zeros((N, N))
        for prev, curr in zip(hist, hist[1:]):
            counts[prev, curr] += 1
        pi_hat = np.zeros((N, N))
        for i in range(N):
            row_sum = counts[i].sum()
            if row_sum > 0:
                pi_hat[i] = counts[i] / row_sum
            else:
                pi_hat[i] = np.ones(N) / N
        self.Pi = 0.80 * self.PI_BASE + 0.20 * pi_hat
        # Row-normalise
        for i in range(N):
            self.Pi[i] /= self.Pi[i].sum()

    # ── Fused estimates ───────────────────────────────────────────────────────

    def fpos(self) -> np.ndarray:
        return sum(self.mu[i] * self.models[i].pos() for i in range(len(self.models)))

    def fvel(self) -> np.ndarray:
        return sum(self.mu[i] * self.models[i].vel() for i in range(len(self.models)))

    def future(self, steps: int, dt: float = DT_DEFAULT) -> np.ndarray:
        return sum(self.mu[i] * self.models[i].predict_pos(steps, dt)
                   for i in range(len(self.models)))

    def dominant_model(self) -> str:
        return self.models[int(np.argmax(self.mu))].name

    def track_quality(self) -> float:
        """0–1 based on recent innovation magnitude."""
        if len(self._innov) < 2:
            return 0.3
        mean_innov = float(np.mean(list(self._innov)[-10:]))
        return float(np.clip(1.0 - mean_innov / 2.0, 0.0, 1.0))

    def rmse(self) -> float:
        if len(self.true_hist) < 2 or len(self.est_hist) < 2:
            return 0.0
        errs = [float(np.linalg.norm(e - t))
                for e, t in zip(self.est_hist, self.true_hist)]
        return float(np.sqrt(np.mean(np.array(errs) ** 2)))
