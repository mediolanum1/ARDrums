import math

class KinematicDepthEstimator:
    def __init__(self):
        """
        Uses Inverse Kinematics (IK) and physical bone lengths to calculate 
        the true 3D depth of joints, bypassing neural network depth guesses.
        """
        self.l_upper = 0.0
        self.l_forearm = 0.0
        self.is_calibrated = False

    def calibrate_exact_lengths(self, upper_arm_m, forearm_m):
        """
        Call this once during calibration (e.g., from a T-Pose).
        
        :param upper_arm_m: True physical length of the upper arm in meters.
        :param forearm_m: True physical length of the forearm in meters.
        """
        self.l_upper = upper_arm_m
        self.l_forearm = forearm_m
        self.is_calibrated = True

    def estimate_chain_z(self, shoulder_px, elbow_px, wrist_px, metric_to_px_scale, mp_elbow_z=0.0, mp_wrist_z=0.0, shoulder_z=0.0):
        """
        Calculates the exact Z depth of the elbow and wrist by measuring how much
        the 2D on-screen lengths have foreshortened compared to the true physical lengths.
        
        :returns: Tuple of (elbow_z, wrist_z) in world coordinates.
        """
        if not self.is_calibrated or metric_to_px_scale <= 0:
            return shoulder_z, shoulder_z # Fallback to flat Z if uncalibrated

        px_to_m = 1.0 / metric_to_px_scale

        # 1. Measure 2D Projected Lengths (what the camera currently sees)
        p_upper_m = math.hypot(elbow_px[0] - shoulder_px[0], elbow_px[1] - shoulder_px[1]) * px_to_m
        p_forearm_m = math.hypot(wrist_px[0] - elbow_px[0], wrist_px[1] - elbow_px[1]) * px_to_m

        # 2. Pythagoras for Upper Arm (Shoulder -> Elbow)
        # max(0, ...) protects against 2D tracking noise making the arm appear longer than reality
        dz_upper = math.sqrt(max(0, self.l_upper**2 - p_upper_m**2))
        
        # 3. Pythagoras for Forearm (Elbow -> Wrist)
        dz_forearm = math.sqrt(max(0, self.l_forearm**2 - p_forearm_m**2))

        # 4. Chain them together
        # In MediaPipe, -Z moves towards the camera. Arms generally point forward
        # .
        elbow_z = shoulder_z - dz_upper

        # {idk about this} this is experimental, basically if wrist is closer then elbwo to body ,then here calc is wrong
        #wrist_z = elbow_z - dz_forearm 

        sign = 1 if mp_wrist_z > mp_elbow_z else -1
        wrist_z = elbow_z + sign * dz_forearm


        return elbow_z, wrist_z