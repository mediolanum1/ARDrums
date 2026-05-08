import math
import time

# State Constants
UP = 0
DOWN = 1

# Thresholds (Tune these based on your webcam distance)
SPEED_THRESHOLD = 0.05
VEL_THRESHOLD = 0.03
COOLDOWN_MS = 200
STATE_CHANGE_FRAME_THRESHOLD = 2

class GestureWristProcessor:
    def __init__(self, label):
        self.label = label
        # Your state machine data
        self.state = UP
        self.peak_dy = 0
        self.state_change_frame = 0
        self.last_hit_time = 0
        self.prev_wrist_px = None
        
        # Spatial/Depth data
        self.z_memory = 0.0
        self.z_offset = 0.0

    def process(self, w_scr, w_wrl, sh_scr, sh_wrl, el_scr, sw_m, kit, cur_time_ms, frame_dims, other_sh_scr):
        w, h = frame_dims
        wrist_px = (w_scr.x * w, w_scr.y * h)
        
        # 1. Scale Ruler: How many pixels wide is the person?
        current_sw_px = math.hypot(sh_scr.x - other_sh_scr.x, sh_scr.y - other_sh_scr.y) * w
        if current_sw_px == 0: current_sw_px = 1
        
        # 2. 2D Motion Calculation
        dx = dy = speed = 0
        norm_dx = norm_dy = norm_speed = 0 # Initialize normalized values
        
        if self.prev_wrist_px is not None:
            dx = wrist_px[0] - self.prev_wrist_px[0]
            dy = wrist_px[1] - self.prev_wrist_px[1]
            
            # NORMALIZING HERE: Turn pixels into "percentage of body width"
            norm_dx = dx / current_sw_px
            norm_dy = dy / current_sw_px
            norm_speed = math.hypot(norm_dx, norm_dy)

        # 3. Occlusion Correction (Depth Estimation)
        dist_ws = math.hypot(w_scr.x - sh_scr.x, w_scr.y - sh_scr.y)
        dist_es = math.hypot(el_scr.x - sh_scr.x, el_scr.y - sh_scr.y)
        
        corrected_z = w_wrl.z 
        is_occluded = dist_ws < 0.15 and dist_es > 0.15
        if is_occluded:
            damp = dist_ws / 0.15
            corrected_z = sh_wrl.z + ((w_wrl.z - sh_wrl.z) * damp)

        raw_z_tared = (corrected_z / sw_m) - self.z_offset
        self.z_memory = (self.z_memory * 0.8) + (raw_z_tared * 0.2)

        # 4. State Machine (Using ONLY Normalized Values)
        hit_detected = None

        if self.state == UP:
            # Use norm_speed
            if norm_speed > SPEED_THRESHOLD and self.state_change_frame > STATE_CHANGE_FRAME_THRESHOLD:
                self.state = DOWN
                self.peak_dy = norm_dy # Store normalized peak
                self.state_change_frame = 0
            else:
                self.state_change_frame += 1

        elif self.state == DOWN:
            # Track normalized peak
            if norm_dy > self.peak_dy:
                self.peak_dy = norm_dy

            # Check thresholds using normalized peak and speed
            if (self.peak_dy > VEL_THRESHOLD and 
                norm_speed > SPEED_THRESHOLD and 
                (cur_time_ms - self.last_hit_time) > COOLDOWN_MS):
                
                # --- GESTURE TRIGGERED ---
                raw_x = w_wrl.x / sw_m
                raw_y = w_wrl.y / sw_m
                
                hit_detected = kit.check_hit(raw_x, raw_y, self.z_memory, cur_time_ms / 1000.0)
                
                self.last_hit_time = cur_time_ms
                self.state_change_frame += 1

            # Use norm_speed for reset
            elif norm_dy < -SPEED_THRESHOLD:
                self.state = UP
                self.state_change_frame = 0

        self.prev_wrist_px = wrist_px
        
        debug_info = {
            "pos_px": (int(wrist_px[0]), int(wrist_px[1])),
            "sh_px": (int(sh_scr.x * w), int(sh_scr.y * h)),
            "is_occluded": is_occluded,
            "z": self.z_memory,
            "state": "DOWN" if self.state == DOWN else "UP",
            "hit": hit_detected,
            "debug_speed": norm_speed # Useful for tuning
        }
        
        return hit_detected, debug_info