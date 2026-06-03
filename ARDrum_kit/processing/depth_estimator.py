import math

class KinematicDepthEstimator:
    def __init__(self):
        self.l_upper = 0.0
        self.l_forearm = 0.0
        self.is_calibrated = False

    def calibrate_exact_lengths(self, upper_arm_m, forearm_m):
        self.l_upper = upper_arm_m
        self.l_forearm = forearm_m
        self.is_calibrated = True

    def estimate_chain_z(self, shoulder_px, elbow_px, wrist_px, metric_to_px_scale, mp_elbow_z=0.0, mp_wrist_z=0.0, mp_shoulder_z=0.0):
        if not self.is_calibrated or metric_to_px_scale <= 0:
            return mp_shoulder_z, mp_shoulder_z 

        px_to_m = 1.0 / metric_to_px_scale

        p_upper_m = math.hypot(elbow_px[0] - shoulder_px[0], elbow_px[1] - shoulder_px[1]) * px_to_m
        p_forearm_m = math.hypot(wrist_px[0] - elbow_px[0], wrist_px[1] - elbow_px[1]) * px_to_m

        dz_upper = math.sqrt(max(0, self.l_upper**2 - p_upper_m**2))
        
        dz_forearm = math.sqrt(max(0, self.l_forearm**2 - p_forearm_m**2))

        elbow_z = mp_shoulder_z - dz_upper

        # {idk about this} this is experimental, basically if wrist is closer then elbwo to body ,then here calc is wrong
        #wrist_z = elbow_z - dz_forearm 

        sign = 1 if mp_wrist_z > mp_elbow_z else -1
        wrist_z = elbow_z + sign * dz_forearm

        return elbow_z, wrist_z
    


    def estimate_chain_z_fusion(self, shoulder_px, elbow_px, wrist_px, metric_to_px_scale, mp_elbow_z=0.0, mp_wrist_z=0.0, mp_shoulder_z=0.0):
            if not self.is_calibrated or metric_to_px_scale <= 0:
                return mp_shoulder_z, mp_shoulder_z 

            px_to_m = 1.0 / metric_to_px_scale

    
            p_upper_m = math.hypot(elbow_px[0] - shoulder_px[0], elbow_px[1] - shoulder_px[1]) * px_to_m
            p_forearm_m = math.hypot(wrist_px[0] - elbow_px[0], wrist_px[1] - elbow_px[1]) * px_to_m
            dz_upper = math.sqrt(max(0, self.l_upper**2 - p_upper_m**2))
            dz_forearm = math.sqrt(max(0, self.l_forearm**2 - p_forearm_m**2))

            
            kinematic_elbow_z = mp_shoulder_z - dz_upper
           
            sign = 1 if mp_wrist_z > mp_elbow_z else -1
            kinematic_wrist_z = kinematic_elbow_z + sign * dz_forearm

    
            upper_ratio = p_upper_m / max(0.001, self.l_upper)
            forearm_ratio = p_forearm_m / max(0.001, self.l_forearm)
            
            conf_upper = max(0.0, min(1.0, (upper_ratio - 0.15) / (0.40 - 0.15)))
            conf_forearm = max(0.0, min(1.0, (forearm_ratio - 0.15) / (0.40 - 0.15)))

            mp_upper_length = math.sqrt(p_upper_m**2 + (mp_elbow_z - mp_shoulder_z)**2)
            if abs(mp_upper_length - self.l_upper) > (self.l_upper * 0.5):
                conf_upper = 1.0  
                
            mp_forearm_length = math.sqrt(p_forearm_m**2 + (mp_wrist_z - mp_elbow_z)**2)
            if abs(mp_forearm_length - self.l_forearm) > (self.l_forearm * 0.5):
                conf_forearm = 1.0 

        
            final_elbow_z = (kinematic_elbow_z * conf_upper) + (mp_elbow_z * (1.0 - conf_upper))
            final_wrist_z = (kinematic_wrist_z * conf_forearm) + (mp_wrist_z * (1.0 - conf_forearm))

            return final_elbow_z, final_wrist_z