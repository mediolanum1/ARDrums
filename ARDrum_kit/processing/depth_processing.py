import time

class _LM:
   # wrapper for MP outputs cuz they are read-only
    __slots__ = ("x", "y", "z", "visibility", "presence")
    def __init__(self, lm, z=None):
        self.x          = lm.x
        self.y          = lm.y
        self.z          = z if z is not None else lm.z
        self.visibility = getattr(lm, "visibility", 1.0)
        self.presence   = getattr(lm, "presence", 1.0)

class DepthManager:
    def __init__(self, frame_width, frame_height):

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.depth_boost_weight = 0.0
        
        self._smoothed_l_sh_z = None
        self._smoothed_r_sh_z = None

        self.last_stats_time = 0.0
        self.STATS_INTERVAL_SEC = 2.0  

    def process_kinematic_depth(self, s_lm, w_lm, anatomical_estimator, metric_to_px_scale):
        w_lm_eff = list(w_lm)
        
        def to_px(lm):
            return (lm.x * self.frame_width, lm.y * self.frame_height)
        
        raw_l_z = w_lm_eff[11].z
        raw_r_z = w_lm_eff[12].z
        
        if self._smoothed_l_sh_z is None:
            self._smoothed_l_sh_z = raw_l_z
            self._smoothed_r_sh_z = raw_r_z
        else:

            self._smoothed_l_sh_z = (self._smoothed_l_sh_z * 0.8) + (raw_l_z * 0.2)
            self._smoothed_r_sh_z = (self._smoothed_r_sh_z * 0.8) + (raw_r_z * 0.2)
    
        l_sh_px = to_px(s_lm[11])
        l_el_px = to_px(s_lm[13])
        l_wr_px = to_px(s_lm[15])
        
        mp_l_elbow_z = w_lm_eff[13].z
        mp_l_wrist_z = w_lm_eff[15].z
        
        l_el_geom_z, l_wr_geom_z = anatomical_estimator.estimate_chain_z(
            l_sh_px, l_el_px, l_wr_px, metric_to_px_scale, mp_elbow_z=mp_l_elbow_z, mp_wrist_z=mp_l_wrist_z, mp_shoulder_z=w_lm_eff[11].z
        )
        
        # this is fingers , idk maybe later delete cuz we dont use them
        l_wrist_dz = l_wr_geom_z - mp_l_wrist_z
        w_lm_eff[13] = _LM(w_lm_eff[13], z=l_el_geom_z)
        w_lm_eff[15] = _LM(w_lm_eff[15], z=l_wr_geom_z)
        for idx in [17, 19, 21]:
            w_lm_eff[idx] = _LM(w_lm_eff[idx], z=w_lm_eff[idx].z + l_wrist_dz)

        r_sh_px = to_px(s_lm[12])
        r_el_px = to_px(s_lm[14])
        r_wr_px = to_px(s_lm[16])
        
        mp_r_elbow_z = w_lm_eff[14].z
        mp_r_wrist_z = w_lm_eff[16].z
        
        r_el_geom_z, r_wr_geom_z = anatomical_estimator.estimate_chain_z(
            r_sh_px, r_el_px, r_wr_px, metric_to_px_scale, mp_elbow_z=mp_r_elbow_z, mp_wrist_z=mp_r_wrist_z, mp_shoulder_z=w_lm_eff[12].z
        )
        # fingers same as above
        r_wrist_dz = r_wr_geom_z - mp_r_wrist_z
        w_lm_eff[14] = _LM(w_lm_eff[14], z=r_el_geom_z)
        w_lm_eff[16] = _LM(w_lm_eff[16], z=r_wr_geom_z)
        for idx in [18, 20, 22]:
            w_lm_eff[idx] = _LM(w_lm_eff[idx], z=w_lm_eff[idx].z + r_wrist_dz)

        current_time = time.time()
        stats_payload = None 
        
        if current_time - self.last_stats_time >= self.STATS_INTERVAL_SEC:
            stats_payload = {
                "timestamp_ms": int(current_time * 1000),
                "left_elbow":  {"mediapipe_z": float(mp_l_elbow_z), "anatomical_z": float(l_el_geom_z), "delta": float(l_el_geom_z - mp_l_elbow_z)},
                "left_wrist":  {"mediapipe_z": float(mp_l_wrist_z), "anatomical_z": float(l_wr_geom_z), "delta": float(l_wr_geom_z - mp_l_wrist_z)},
                "right_elbow": {"mediapipe_z": float(mp_r_elbow_z), "anatomical_z": float(r_el_geom_z), "delta": float(r_el_geom_z - mp_r_elbow_z)},
                "right_wrist": {"mediapipe_z": float(mp_r_wrist_z), "anatomical_z": float(r_wr_geom_z), "delta": float(r_wr_geom_z - mp_r_wrist_z)}
            }
            self.last_stats_time = current_time


#       w_lm_eff = self._apply_drumming_posture_boost(s_lm, w_lm_eff)

        return w_lm_eff, stats_payload

    def _apply_drumming_posture_boost(self, s_lm, w_lm_eff):
        wl, wr = w_lm_eff[15], w_lm_eff[16]

        y_sh_avg = (w_lm_eff[11].y + w_lm_eff[12].y) / 2.0
        y_hip_avg = (w_lm_eff[23].y + w_lm_eff[24].y) / 2.0
        torso_height = y_hip_avg - y_sh_avg

        y_stomach_top    = y_sh_avg + (torso_height * 0.4)
        y_stomach_bottom = y_hip_avg

        torso_x_min = min(s_lm[11].x, s_lm[12].x)
        torso_x_max = max(s_lm[11].x, s_lm[12].x)
        shoulder_w  = torso_x_max - torso_x_min
        h_padding   = shoulder_w * 0.10  

        l_in_torso = torso_x_min - h_padding < s_lm[15].x < torso_x_max + h_padding
        r_in_torso = torso_x_min - h_padding < s_lm[16].x < torso_x_max + h_padding

        condition_met = (
            y_stomach_top < wl.y < y_stomach_bottom and
            y_stomach_top < wr.y < y_stomach_bottom and
            wl.z < -0.2 and wr.z < -0.2 and
            l_in_torso and r_in_torso
        )

        _BOOST_RISE = 0.25   
        _BOOST_FALL = 0.05   

        if condition_met:
            self.depth_boost_weight = min(1.0, self.depth_boost_weight + _BOOST_RISE)
        else:
            self.depth_boost_weight = max(0.0, self.depth_boost_weight - _BOOST_FALL)

        if self.depth_boost_weight > 1e-3:
            effective_boost = -0.08 * self.depth_boost_weight
            w_lm_eff[15] = _LM(wl, z=wl.z + effective_boost)
            w_lm_eff[16] = _LM(wr, z=wr.z + effective_boost)
            
        
            for idx in [17, 19, 21, 18, 20, 22]:
                w_lm_eff[idx] = _LM(w_lm_eff[idx], z=w_lm_eff[idx].z + effective_boost)

        return w_lm_eff