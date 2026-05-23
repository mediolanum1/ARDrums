"""
kalman_wrist.py
---------------
Constant-velocity Kalman filter for MediaPipe 3-D wrist coordinates.

State vector  : [x, y, z, vx, vy, vz]   (6 × 1)
Measurement   : [x, y, z]               (3 × 1)

Drop-in usage inside GestureWristProcessor:

    from kalman_wrist import WristKalman
    self._kf = WristKalman()

    # Replace the raw/smoothed z passthrough with:
    kx, ky, kz = self._kf.update(w_wrl.x, w_wrl.y, w_wrl.z)
    curr_3d_coords = (kx, ky, kz)
"""

import cv2
import numpy as np


class WristKalman:
    """
    6-state, 3-measurement Kalman filter.

    Tune the two noise parameters to your setup:
      process_noise   – how much you trust the constant-velocity model.
                        Raise if the hand accelerates sharply (fast hits).
      measurement_noise – how much you trust MediaPipe's raw output.
                        Raise if MediaPipe is noisy; lower if it is stable.
    """

    def __init__(self, dt: float = 1 / 30,
                 process_noise: float = 1e-2,
                 measurement_noise: float = 1e-1):

        # cv2.KalmanFilter(dynamParams, measureParams)
        self._kf = cv2.KalmanFilter(6, 3)

        # ── Transition matrix  (constant-velocity model) ──────────────────
        # x_new = x + vx*dt,  vx_new = vx   (same for y, z)
        F = np.eye(6, dtype=np.float32)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self._kf.transitionMatrix = F

        # ── Measurement matrix  H: maps state → [x, y, z] ────────────────
        H = np.zeros((3, 6), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        self._kf.measurementMatrix = H

        # ── Process noise covariance  Q ───────────────────────────────────
        # Higher value → filter reacts faster but smooths less.
        self._kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise

        # ── Measurement noise covariance  R ───────────────────────────────
        # Higher value → filter trusts measurements less (smoother, more lag).
        # Z gets a larger value because MediaPipe's depth estimate is noisier.
        R = np.eye(3, dtype=np.float32) * measurement_noise
        R[2, 2] = measurement_noise * 5      # extra distrust for Z
        self._kf.measurementNoiseCov = R

        # ── Initial error covariance  P ───────────────────────────────────
        self._kf.errorCovPost = np.eye(6, dtype=np.float32)

        self._initialised = False

    # ─────────────────────────────────────────────────────────────────────────

    def update(self, x: float, y: float, z: float):
        """
        Feed one MediaPipe measurement and return the filtered (x, y, z).

        On the very first call the state is seeded from the measurement so
        there is no initial transient.
        """
        measurement = np.array([[x], [y], [z]], dtype=np.float32)

        if not self._initialised:
            # Seed position; leave velocity at zero.
            self._kf.statePost = np.array(
                [x, y, z, 0.0, 0.0, 0.0], dtype=np.float32
            ).reshape(6, 1)
            self._initialised = True

        self._kf.predict()
        corrected = self._kf.correct(measurement)   # (6, 1)

        return float(corrected[0]), float(corrected[1]), float(corrected[2])

    def predict_only(self):
        """
        Advance the filter one frame WITHOUT a measurement.
        Call this when MediaPipe drops a frame entirely — the filter
        will extrapolate position from the last known velocity.
        Returns the predicted (x, y, z).
        """
        if not self._initialised:
            return 0.0, 0.0, 0.0
        predicted = self._kf.predict()
        return float(predicted[0]), float(predicted[1]), float(predicted[2])

    def reset(self):
        """Force re-initialisation on the next update() call."""
        self._initialised = False