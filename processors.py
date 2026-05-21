import math

UP = 0
DOWN = 1

SPEED_THRESHOLD              = 0.015
COOLDOWN_MS                  = 100
STATE_CHANGE_FRAME_THRESHOLD = 1
MIN_DOWNWARD_MOTION          = 0.01
MIN_HORIZONTAL_MOTION        = 0.04
MIN_UPWARD_MOTION            = -0.01
STALL_RESET_FRAME_THRESHOLD  = 12
STALL_SPEED_THRESHOLD        = 0.008


class GestureWristProcessor:
    def __init__(self, label):
        self.label = label
        self.state = UP
        self.state_change_frame = 0
        self.last_hit_time = 0

        self.prev_wrist_px = None
        self.prev_3d_coords = None

        self.smooth_norm_speed = 0.0

    def process(self, w_scr, w_wrl, sh_scr, sh_wrl, el_scr, sw_m, kit, cur_time_ms, frame_dims, other_sh_scr):
        w, h = frame_dims
        wrist_px = (w_scr.x * w, w_scr.y * h)

        # Scale ruler (current shoulder width in pixels) — used only for 2-D motion normalisation
        current_sw_px = math.hypot(sh_scr.x - other_sh_scr.x, sh_scr.y - other_sh_scr.y) * w
        if current_sw_px == 0:
            current_sw_px = 1

        # ── Raw MediaPipe world coords — no modification ──────────────────────
        curr_3d_coords = (w_wrl.x, w_wrl.y, w_wrl.z)

        # ── 2-D screen-space motion (for gesture state machine only) ─────────
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

        # ── State machine ─────────────────────────────────────────────────────
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
                    downward_motion > 0 and
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

        # ── Update per-frame memory ───────────────────────────────────────────
        self.prev_wrist_px  = wrist_px
        self.prev_3d_coords = curr_3d_coords

        debug_info = {
            "pos_px":      (int(wrist_px[0]), int(wrist_px[1])),
            "sh_px":       (int(sh_scr.x * w), int(sh_scr.y * h)),
            "z":           w_wrl.z,
            "state":       "DOWN" if self.state == DOWN else "UP",
            "hit":         hit_detected,
            "debug_speed": self.smooth_norm_speed,
            # Raw 3-D world position straight from MediaPipe
            "norm_3d":     curr_3d_coords,
        }

        return hit_detected, debug_info