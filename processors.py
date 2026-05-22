import math

UP = 0
DOWN = 1

SPEED_THRESHOLD            = 0.015
COOLDOWN_MS                = 100
STATE_CHANGE_FRAME_THRESHOLD = 1
MIN_DOWNWARD_MOTION        = 0.01
MIN_HORIZONTAL_MOTION      = 0.04
MIN_UPWARD_MOTION          = -0.01
STALL_RESET_FRAME_THRESHOLD = 12
STALL_SPEED_THRESHOLD      = 0.008


class GestureWristProcessor:
    def __init__(self, label):
        self.label = label
        self.state = UP
        self.state_change_frame = 0
        self.last_hit_time = 0

        self.prev_wrist_px = None
        self.prev_3d_coords = None

        self.z_memory = 0.0
        self.z_offset = 0.0
        self.smooth_norm_speed = 0.0

    def process(self, w_scr, w_wrl, sh_scr, sh_wrl, el_scr, sw_m, kit, cur_time_ms, frame_dims, other_sh_scr):
        w, h = frame_dims
        wrist_px = (w_scr.x * w, w_scr.y * h)

        # 1. Scale ruler (current shoulder width in pixels)
        current_sw_px = math.hypot(sh_scr.x - other_sh_scr.x, sh_scr.y - other_sh_scr.y) * w
        if current_sw_px == 0:
            current_sw_px = 1

        # 2. Depth / occlusion correction (2.5D hybrid)
        dist_ws = math.hypot(w_scr.x - sh_scr.x, w_scr.y - sh_scr.y)
        dist_es = math.hypot(el_scr.x - sh_scr.x, el_scr.y - sh_scr.y)

 
 

        # Heuristic 2.5D: estimate depth from 2-D projected arm length.
        # Projected arm length in shoulder-width units (pixel-normalised)
        proj_arm_px = math.hypot(w_scr.x - el_scr.x, w_scr.y - el_scr.y) * w
        proj_len_norm = proj_arm_px / max(1.0, current_sw_px)

        # Typical adult forearm+hand length expressed in shoulder-width units (tunable)
        ARM_LEN_WORLD = 0.65
        if proj_len_norm > ARM_LEN_WORLD:
            proj_len_norm = ARM_LEN_WORLD

        # world-derived z (rotated) in shoulder-width units
        world_z = w_wrl.z

        # z from 2D: magnitude from arm projection, but sign from world estimate
        z_magnitude = math.sqrt(max(0.0, ARM_LEN_WORLD * ARM_LEN_WORLD - proj_len_norm * proj_len_norm))
        z_2d = -z_magnitude if world_z < 0 else z_magnitude

        # occlusion confidence: if wrist is occluded, prefer 2D estimate
        is_occluded = dist_ws < 0.15 and dist_es > 0.15
        if is_occluded:
            chosen_z = z_2d
        else:
            # blend world and 2D estimates for stability
            chosen_z = 0.5 * world_z + 0.5 * z_2d

        # smooth temporal memory
        self.z_memory = (self.z_memory * 0.8) + (chosen_z * 0.2)

        # 3D position in shoulder-width normalised units
        # X uses rotated x; Y uses body-relative y
        
        curr_3d_coords = (w_wrl.x, w_wrl.y, self.z_memory)

        # 3. 2-D motion
        norm_dx = 0.0
        norm_dy = 0.0
        raw_norm_speed = 0.0
        if self.prev_wrist_px is not None:
            dx = wrist_px[0] - self.prev_wrist_px[0]
            dy = wrist_px[1] - self.prev_wrist_px[1]
            norm_dx = dx / current_sw_px
            norm_dy = dy / current_sw_px
            raw_norm_speed = math.hypot(norm_dx, norm_dy)
            self.smooth_norm_speed = (self.smooth_norm_speed * 0.3) + (raw_norm_speed * 0.7)

        downward_motion   = norm_dy
        horizontal_motion = abs(norm_dx)
        upward_motion     = norm_dy

        # 4. State machine
        hit_detected = None

        if self.state == UP:
            if raw_norm_speed > SPEED_THRESHOLD and (
                downward_motion > MIN_DOWNWARD_MOTION or
                horizontal_motion > MIN_HORIZONTAL_MOTION
            ):
                self.state_change_frame += 1
                if self.state_change_frame > STATE_CHANGE_FRAME_THRESHOLD:
                    self.state = DOWN
                    self.state_change_frame = 0
            else:
                self.state_change_frame = 0

        elif self.state == DOWN:
            if (self.smooth_norm_speed > SPEED_THRESHOLD and
                    self.prev_3d_coords is not None and
                    (downward_motion > 0 or horizontal_motion > MIN_HORIZONTAL_MOTION) and
                    (cur_time_ms - self.last_hit_time) > COOLDOWN_MS):

                hit_detected = kit.check_line_intersection(
                    curr_3d_coords,
                    cur_time_ms / 1000.0
                )

                if hit_detected:
                    self.last_hit_time = cur_time_ms
                    self.state = UP

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

        # Update per-frame memory
        self.prev_wrist_px  = wrist_px
        self.prev_3d_coords = curr_3d_coords

        debug_info = {
            "pos_px":      (int(wrist_px[0]), int(wrist_px[1])),
            "sh_px":       (int(sh_scr.x * w), int(sh_scr.y * h)),
            "is_occluded": is_occluded,
            "z":           self.z_memory,
            "z_2d":        z_2d,
            "proj_len":    proj_len_norm,
            "world_z":     world_z,
            "state":       "DOWN" if self.state == DOWN else "UP",
            "hit":         hit_detected,
            "debug_speed": self.smooth_norm_speed,
            # Normalised 3-D position in shoulder-width units.
            # X/Y match drum center[0]/center[1] — used directly by the POV panel.
            "norm_3d":     curr_3d_coords,
        }

        return hit_detected, debug_info