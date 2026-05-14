import math

UP = 0
DOWN = 1

SPEED_THRESHOLD = 0.03
COOLDOWN_MS = 100
STATE_CHANGE_FRAME_THRESHOLD = 5

class GestureWristProcessor:
    def __init__(self, label):
        self.label = label
        self.state = UP
        self.state_change_frame = 0
        self.last_hit_time = 0
        self.forearm_length_m = 0.25   # fallback default ~25cm
        self.upper_arm_length_m = 0.30
        # Memory for 2D and 3D previous frames
        self.prev_wrist_px = None
        self.prev_3d_coords = None 
        
        self.z_memory = 0.0
        self.z_offset = 0.0
        self.smooth_norm_speed = 0.0
    def process(self, w_scr, w_wrl, sh_scr, sh_wrl, el_scr, el_wrl, sw_m, kit, cur_time_ms, frame_dims, other_sh_scr):
        #                                          ^^^^^^ add el_wrl
        w, h = frame_dims
        wrist_px = (w_scr.x * w, w_scr.y * h)

        current_sw_px = math.hypot(sh_scr.x - other_sh_scr.x, sh_scr.y - other_sh_scr.y) * w
        if current_sw_px == 0: current_sw_px = 1

        # --- Reliability score (0 = arm at camera, 1 = arm perpendicular) ---
        reliability = self._foreshortening_reliability(w_scr, el_scr, sw_m, current_sw_px)

        # --- Existing occlusion + tare logic (unchanged) ---
        dist_ws = math.hypot(w_scr.x - sh_scr.x, w_scr.y - sh_scr.y)
        dist_es = math.hypot(el_scr.x - sh_scr.x, el_scr.y - sh_scr.y)
        corrected_z = w_wrl.z
        is_occluded = dist_ws < 0.15 and dist_es > 0.15
        if is_occluded:
            damp = dist_ws / 0.15
            corrected_z = sh_wrl.z + ((w_wrl.z - sh_wrl.z) * damp)

        mediapipe_z_tared = (corrected_z / sw_m) - self.z_offset

        # --- Arm-length estimated Z ---
        estimated_z_raw = self._estimate_wrist_z_from_arm(w_wrl, el_wrl, sw_m)
        if estimated_z_raw is not None:
            estimated_z_tared = estimated_z_raw - self.z_offset
            # Blend: low reliability → trust arm-length estimate more
            blended_z = reliability * mediapipe_z_tared + (1.0 - reliability) * estimated_z_tared
        else:
            blended_z = mediapipe_z_tared   # fallback if math breaks

        raw_z_tared = blended_z
        self.z_memory = (self.z_memory * 0.8) + (raw_z_tared * 0.2)
        
        # Calculate current 3D position
        raw_x = w_wrl.x / sw_m
        raw_y = w_wrl.y / sw_m
        curr_3d_coords = (raw_x, raw_y, self.z_memory)

        # 3. 2D Motion Calculation
        norm_dy = 0
        if self.prev_wrist_px is not None:
            dx = wrist_px[0] - self.prev_wrist_px[0]
            dy = wrist_px[1] - self.prev_wrist_px[1]
            
            norm_dx = dx / current_sw_px
            norm_dy = dy / current_sw_px
            raw_norm_speed = math.hypot(norm_dx, norm_dy)
            
            # Smooth the speed to prevent frame jitter
            self.smooth_norm_speed = (self.smooth_norm_speed * 0.5) + (raw_norm_speed * 0.5)

        # 4. JOINT ESTIMATION STATE MACHINE
        hit_detected = None

        if self.state == UP:
            # Intent Check: Moving fast and downwards?
            if self.smooth_norm_speed > SPEED_THRESHOLD and (norm_dy > 0 or norm_dx>0):
                self.state_change_frame += 1
                if self.state_change_frame > STATE_CHANGE_FRAME_THRESHOLD:
                    self.state = DOWN
                    self.state_change_frame = 0
            else:
                self.state_change_frame = 0

        elif self.state == DOWN:
            # --- JOINT ESTIMATION TRIGGER ---
            # 1. We are in DOWN state (Intent)
            # 2. We are moving fast enough (Speed)
            # 3. We draw a line from Prev Frame to Curr Frame (Spatial Raycast)
            
            if (self.smooth_norm_speed > SPEED_THRESHOLD and 
                self.prev_3d_coords is not None and 
                (cur_time_ms - self.last_hit_time) > COOLDOWN_MS):
                
                # Check line intersection!
                hit_detected = kit.check_line_intersection(
                    self.prev_3d_coords, 
                    curr_3d_coords, 
                    cur_time_ms / 1000.0
                )
                
                if hit_detected:
                    self.last_hit_time = cur_time_ms

            # Reset State: Hand is moving back up
            if norm_dy < 0:
                self.state_change_frame += 1
                if self.state_change_frame > STATE_CHANGE_FRAME_THRESHOLD:
                    self.state = UP
                    self.state_change_frame = 0
            else:
                self.state_change_frame = 0
        # Update Memory for the next frame's "Line"
        self.prev_wrist_px = wrist_px
        self.prev_3d_coords = curr_3d_coords
        
        debug_info = {
            "pos_px": (int(wrist_px[0]), int(wrist_px[1])),
            "sh_px": (int(sh_scr.x * w), int(sh_scr.y * h)),
            "is_occluded": is_occluded,
            "z": self.z_memory,
            "state": "DOWN" if self.state == DOWN else "UP",
            "hit": hit_detected,
            "debug_speed": self.smooth_norm_speed
        }
        
        return hit_detected, debug_info
    
    def _estimate_wrist_z_from_arm(self, w_wrl, el_wrl, sw_m):
        """
        Sphere intersection: wrist lies on sphere of radius forearm_length
        centered at elbow. Solve for Z using known XY positions.

        Returns estimated Z (normalized by sw_m, tared), or None if unsolvable.
        """
        fl_n = self.forearm_length_m / sw_m   # forearm length in shoulder-width units

        # Normalized XY displacement wrist→elbow
        dx = (w_wrl.x - el_wrl.x) / sw_m
        dy = (w_wrl.y - el_wrl.y) / sw_m

        xy_dist_sq = dx**2 + dy**2

        # If XY distance already exceeds arm length, model has gone haywire
        if xy_dist_sq > fl_n**2:
            return None

        z_component = math.sqrt(fl_n**2 - xy_dist_sq)

        # Wrist is typically in front of (more negative Z than) elbow during a drum hit
        el_z_n = el_wrl.z / sw_m
        return el_z_n - z_component   # raw, not yet tared

    def _foreshortening_reliability(self, w_scr, el_scr, sw_m, current_sw_px):
        """
        Returns 0.0 (arm fully at camera = MediaPipe Z unreliable)
        to 1.0 (arm fully perpendicular = MediaPipe Z reliable).
        """
        forearm_2d_px = math.hypot(
            (w_scr.x - el_scr.x),
            (w_scr.y - el_scr.y)
        ) * current_sw_px                                  # pixel length of visible forearm
        forearm_full_px = (self.forearm_length_m / sw_m) * current_sw_px  # expected full length in px
        if forearm_full_px < 1:
            return 1.0
        return min(forearm_2d_px / forearm_full_px, 1.0)  # cosine of angle to camera axis
