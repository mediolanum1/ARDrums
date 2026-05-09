import math

UP = 0
DOWN = 1

SPEED_THRESHOLD = 0.025
COOLDOWN_MS = 100
STATE_CHANGE_FRAME_THRESHOLD = 2
MIN_DOWNWARD_MOTION = 0.01
MIN_HORIZONTAL_MOTION = 0.04
MIN_UPWARD_MOTION = -0.01
STALL_RESET_FRAME_THRESHOLD = 12
STALL_SPEED_THRESHOLD = 0.008

class GestureWristProcessor:
    def __init__(self, label):
        self.label = label
        self.state = UP
        self.state_change_frame = 0
        self.last_hit_time = 0
        
        # Memory for 2D and 3D previous frames
        self.prev_wrist_px = None
        self.prev_3d_coords = None 
        
        self.z_memory = 0.0
        self.z_offset = 0.0
        self.smooth_norm_speed = 0.0

    def process(self, w_scr, w_wrl, sh_scr, sh_wrl, el_scr, sw_m, kit, cur_time_ms, frame_dims, other_sh_scr):
        w, h = frame_dims
        wrist_px = (w_scr.x * w, w_scr.y * h)
        
        # 1. Scale Ruler
        current_sw_px = math.hypot(sh_scr.x - other_sh_scr.x, sh_scr.y - other_sh_scr.y) * w
        if current_sw_px == 0: current_sw_px = 1
        
        # 2. Depth Math (Occlusion & Tare)
        dist_ws = math.hypot(w_scr.x - sh_scr.x, w_scr.y - sh_scr.y)
        dist_es = math.hypot(el_scr.x - sh_scr.x, el_scr.y - sh_scr.y)
        
        corrected_z = w_wrl.z 
        is_occluded = dist_ws < 0.15 and dist_es > 0.15
        if is_occluded:
            damp = dist_ws / 0.15
            corrected_z = sh_wrl.z + ((w_wrl.z - sh_wrl.z) * damp)

        raw_z_tared = (corrected_z / sw_m) - self.z_offset
        self.z_memory = (self.z_memory * 0.8) + (raw_z_tared * 0.2)
        
        # Calculate current 3D position
        raw_x = w_wrl.x / sw_m
        raw_y = w_wrl.y / sw_m
        curr_3d_coords = (raw_x, raw_y, self.z_memory)

        # 3. 2D Motion Calculation
        norm_dx = 0.0
        norm_dy = 0.0
        raw_norm_speed = 0.0
        if self.prev_wrist_px is not None:
            dx = wrist_px[0] - self.prev_wrist_px[0]
            dy = wrist_px[1] - self.prev_wrist_px[1]
            
            norm_dx = dx / current_sw_px
            norm_dy = dy / current_sw_px
            raw_norm_speed = math.hypot(norm_dx, norm_dy)
            
            # Smooth the speed to prevent frame jitter
            self.smooth_norm_speed = (self.smooth_norm_speed * 0.3) + (raw_norm_speed * 0.7)

        downward_motion = norm_dy
        horizontal_motion = abs(norm_dx)
        upward_motion = norm_dy

        # 4. JOINT ESTIMATION STATE MACHINE
        hit_detected = None

        if self.state == UP:
            # Intent Check: moving fast and either downward or horizontally strong enough
            if raw_norm_speed > SPEED_THRESHOLD and (
                downward_motion > MIN_DOWNWARD_MOTION or horizontal_motion > MIN_HORIZONTAL_MOTION
            ):
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

            # Reset State: Hand is moving back up or has stalled after a hit
            if upward_motion < MIN_UPWARD_MOTION:
                self.state_change_frame += 1
                if self.state_change_frame > STATE_CHANGE_FRAME_THRESHOLD:
                    self.state = UP
                    self.state_change_frame = 0
            elif self.smooth_norm_speed < STALL_SPEED_THRESHOLD:
                self.state_change_frame += 1
                if self.state_change_frame > STALL_RESET_FRAME_THRESHOLD:
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