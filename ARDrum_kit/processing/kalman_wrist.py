import cv2
import numpy as np


class WristKalman:
    def __init__(self, dt: float = 1 / 30,
                 process_noise: float = 1e-2,
                 measurement_noise: float = 1e-1):
        ''' this is kalman with6 hidden vars that doesnt account for acceleration, if new one feels wierd return to this
        self._kf = cv2.KalmanFilter(6, 3)
        F = np.eye(6, dtype=np.float32)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
       
        self._kf.transitionMatrix = F
        '''
        self._kf = cv2.KalmanFilter(9, 3)
        F = np.eye(9, dtype=np.float32)

        # Position updates (x + v*dt + 0.5*a*dt^2)
        F[0, 3] = dt; F[0, 6] = 0.5 * dt**2  # X-axis
        F[1, 4] = dt; F[1, 7] = 0.5 * dt**2  # Y-axis
        F[2, 5] = dt; F[2, 8] = 0.5 * dt**2  # Z-axis

        # Velocity updates (v + a*dt)
        F[3, 6] = dt  # X-axis velocity
        F[4, 7] = dt  # Y-axis velocity
        F[5, 8] = dt  # Z-axis velocity

        self._kf.transitionMatrix = F
        H = np.zeros((3, 6), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        self._kf.measurementMatrix = H
        self._kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        R = np.eye(3, dtype=np.float32) * measurement_noise
        R[2, 2] = measurement_noise * 2
        self._kf.measurementNoiseCov = R
        self._kf.errorCovPost = np.eye(6, dtype=np.float32)
        self._initialised = False

    def update(self, x: float, y: float, z: float):
        measurement = np.array([[x], [y], [z]], dtype=np.float32)
        if not self._initialised:
            self._kf.statePost = np.array(
                [x, y, z, 0.0, 0.0, 0.0], dtype=np.float32
            ).reshape(6, 1)
            self._initialised = True
        self._kf.predict()
        corrected = self._kf.correct(measurement)  
        return float(corrected[0]), float(corrected[1]), float(corrected[2])

    def predict_only(self):
        if not self._initialised:
            return 0.0, 0.0, 0.0
        predicted = self._kf.predict()
        return float(predicted[0]), float(predicted[1]), float(predicted[2])

    def reset(self):
        self._initialised = False