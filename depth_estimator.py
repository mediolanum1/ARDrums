import math

class KinematicDepthEstimator:
    def __init__(self):
        self.l_upper = 0.0
        self.l_forearm = 0.0
        self.is_calibrated = False

    def calibrate_exact_lengths(self, upper_arm_m, forearm_m):
        """
        Call this once during calibration.
        upper_arm_m and forearm_m are the true physical lengths in meters.
        """
        self.l_upper = upper_arm_m
        self.l_forearm = forearm_m
        self.is_calibrated = True

    def estimate_chain_z(self, shoulder_px, elbow_px, wrist_px, metric_to_px_scale, shoulder_z=0.0):
        """
        Calculates the exact Z depth of the elbow and wrist.
        """
        if not self.is_calibrated or metric_to_px_scale <= 0:
            return shoulder_z, shoulder_z # Fallback

        px_to_m = 1.0 / metric_to_px_scale

        # 1. Measure 2D Projected Lengths (what the camera currently sees)
        p_upper_m = math.hypot(elbow_px[0] - shoulder_px[0], elbow_px[1] - shoulder_px[1]) * px_to_m
        p_forearm_m = math.hypot(wrist_px[0] - elbow_px[0], wrist_px[1] - elbow_px[1]) * px_to_m

        # 2. Pythagoras for Upper Arm (Shoulder -> Elbow)
        # Using max(0, ...) protects against noise making the 2D distance slightly larger than the known 3D length
        dz_upper = math.sqrt(max(0, self.l_upper**2 - p_upper_m**2))
        
        # 3. Pythagoras for Forearm (Elbow -> Wrist)
        dz_forearm = math.sqrt(max(0, self.l_forearm**2 - p_forearm_m**2))

        # 4. Chain them together
        # Note: In MediaPipe, Z is negative moving towards the camera. 
        # For a drumming app, arms generally point forward.
        elbow_z = shoulder_z - dz_upper
        wrist_z = elbow_z - dz_forearm

        return elbow_z, wrist_z