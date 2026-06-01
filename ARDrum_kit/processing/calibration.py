import math
import time

class CalibrationManager:
    def __init__(self, frame_width, frame_height, focal_length, target_frames=5):
        """
        Manages the T-pose calibration phase, calculating the pixel-to-metric scale,
        estimating user distance, and configuring the anatomical depth estimator.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.focal_length = focal_length
        self.target_frames = target_frames

        # State Variables
        self.is_calibrated = False
        self.fixed_sw_m = 1.0
        self.metric_to_px_scale = 1.0
        self.calibrated_distance_m = None
        
        self.error_msg = ""
        self.start_time = time.time()
        self.COUNTDOWN_SECONDS = 5

        # Data Buffers
        self._sw_m_list = []
        self._sw_px_list = []
        self._p_upper_px_list = []
        self._p_forearm_px_list = []

    def reset(self):
        """Clears buffers and resets the countdown timer. Used when calibration fails."""
        self.is_calibrated = False
        self.start_time = time.time()
        
        self._sw_m_list.clear()
        self._sw_px_list.clear()
        self._p_upper_px_list.clear()
        self._p_forearm_px_list.clear()

    def update(self, s_lm, w_lm, depth_estimator):
        """
        Called every frame. Handles the countdown, takes measurements if the 
        countdown is finished, and applies them to the depth_estimator.
        
        :returns: True if calibration is complete, False otherwise.
        """
        if self.is_calibrated:
            return True

        # Wait for the countdown to finish before sampling frames
        elapsed = time.time() - self.start_time
        if elapsed < self.COUNTDOWN_SECONDS:
            return False 

        l_sh_w, r_sh_w = w_lm[11], w_lm[12]
        l_sh_s, r_sh_s = s_lm[11], s_lm[12]
        
        # If shoulders aren't clearly visible, skip this frame
        if l_sh_s.visibility <= 0.5 or r_sh_s.visibility <= 0.5:
            return False

        # 1. World shoulder width (meters)
        cur_sw_m = math.sqrt(
            (l_sh_w.x - r_sh_w.x) ** 2 +
            (l_sh_w.y - r_sh_w.y) ** 2 +
            (l_sh_w.z - r_sh_w.z) ** 2
        )

        # 2. Screen shoulder width (pixels)
        cur_sw_px = math.hypot(
            (l_sh_s.x - r_sh_s.x) * self.frame_width,
            (l_sh_s.y - r_sh_s.y) * self.frame_height,
        )

        # 3. Kinematic pixel lengths (Average Left and Right to remove bias)
        l_sh_px = (s_lm[11].x * self.frame_width, s_lm[11].y * self.frame_height)
        l_el_px = (s_lm[13].x * self.frame_width, s_lm[13].y * self.frame_height)
        l_wr_px = (s_lm[15].x * self.frame_width, s_lm[15].y * self.frame_height)

        r_sh_px = (s_lm[12].x * self.frame_width, s_lm[12].y * self.frame_height)
        r_el_px = (s_lm[14].x * self.frame_width, s_lm[14].y * self.frame_height)
        r_wr_px = (s_lm[16].x * self.frame_width, s_lm[16].y * self.frame_height)

        l_upper_px   = math.hypot(l_el_px[0] - l_sh_px[0], l_el_px[1] - l_sh_px[1])
        l_forearm_px = math.hypot(l_wr_px[0] - l_el_px[0], l_wr_px[1] - l_el_px[1])

        r_upper_px   = math.hypot(r_el_px[0] - r_sh_px[0], r_el_px[1] - r_sh_px[1])
        r_forearm_px = math.hypot(r_wr_px[0] - r_el_px[0], r_wr_px[1] - r_el_px[1])

        p_upper_px   = (l_upper_px + r_upper_px) / 2.0
        p_forearm_px = (l_forearm_px + r_forearm_px) / 2.0

        # 4. Store them
        self._p_upper_px_list.append(p_upper_px)
        self._p_forearm_px_list.append(p_forearm_px)
        self._sw_m_list.append(cur_sw_m)
        self._sw_px_list.append(cur_sw_px)
        print(f"[CAL] Frame {len(self._sw_m_list)}/{self.target_frames} captured.")

        # 5. Check if we have enough frames to finalize
        if len(self._sw_m_list) >= self.target_frames:
            self.fixed_sw_m = sum(self._sw_m_list) / self.target_frames
            avg_sw_px       = sum(self._sw_px_list) / self.target_frames
            avg_upper_px    = sum(self._p_upper_px_list) / self.target_frames
            avg_forearm_px  = sum(self._p_forearm_px_list) / self.target_frames
            
            # Establish the baseline scale
            self.metric_to_px_scale = avg_sw_px / self.fixed_sw_m if self.fixed_sw_m > 0 else 1.0

            # Convert to meters and send to the depth estimator
            avg_upper_arm_m = avg_upper_px / self.metric_to_px_scale
            avg_forearm_m = avg_forearm_px / self.metric_to_px_scale
            depth_estimator.calibrate_exact_lengths(avg_upper_arm_m, avg_forearm_m)

            # Validate the camera distance
            if avg_sw_px > 0 and self.fixed_sw_m > 0:
                self.calibrated_distance_m = (self.focal_length * self.fixed_sw_m) / avg_sw_px
                if self.calibrated_distance_m < 1.0:
                    print(f"[CAL] FAILED: Distance {self.calibrated_distance_m:.2f}m is < 1.0m. Restarting...")
                    self.error_msg = "Please stand at least 1 meter away from camera"
                    self.reset()
                    return False
                    
            print(f"[CAL] DONE. Avg sw={self.fixed_sw_m:.3f} m  dist≈{self.calibrated_distance_m:.2f} m")
            self.error_msg = ""
            self.is_calibrated = True

        return self.is_calibrated

    def get_ui_text(self):
        """Returns tuples of text to render on the screen during the calibration phase."""
        if self.is_calibrated:
            return None, None
            
        elapsed = time.time() - self.start_time
        if elapsed < self.COUNTDOWN_SECONDS:
            rem = int(self.COUNTDOWN_SECONDS - elapsed)
            return f"READY IN: {rem}", self.error_msg
        else:
            return "HOLD T-POSE...", self.error_msg

    def get_current_distance(self, current_sw_px):
        """Estimate live distance to user (metres) using pinhole camera model."""
        if not self.is_calibrated or current_sw_px <= 0 or self.fixed_sw_m <= 0:
            return None
        return (self.focal_length * self.fixed_sw_m) / current_sw_px

    def get_live_metric_to_px_scale(self, current_sw_px):
        """Return the current pixel-to-meter scale using the live shoulder width."""
        if not self.is_calibrated or current_sw_px <= 0 or self.fixed_sw_m <= 0:
            return self.metric_to_px_scale
        return current_sw_px / self.fixed_sw_m