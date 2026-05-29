import math

from kalman_wrist import WristKalman
from typing import Optional
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
        # In processors.py __init__ — one new threshold
        self.MIN_ARM_EXTENSION_M = 0.32   # tweak: ~28 cm shoulder-to-wrist minimum
        self.prev_wrist_px   = None
        self.prev_3d_coords  = None
        self.smooth_norm_speed = 0.0
        self.WORLD_Y_STRIKE_THRESHOLD = 0.003   # ~8 mm downward in world metres

        # ── Kalman filter for 3-D wrist position ─────────────────────────
        # process_noise   : raise if you swing fast and the filter lags behind
        # measurement_noise: raise if MediaPipe is noisy on your camera
        self._kf = WristKalman(
            dt=1 / 30,
            process_noise=1e-2,
            measurement_noise=1e-1,
        )
        self._mediapipe_missing_frames = 0
        # How many consecutive missing frames before we reset the filter.
        # 6 frames (~0.2 s at 30 fps) is generous enough to survive fast
        # swings while not letting a drifted prediction run forever.
        self._MAX_MISSING_FRAMES = 6

    # ─────────────────────────────────────────────────────────────────────

    def process(self, w_scr, w_wrl, sh_scr, sh_wrl, el_scr,
                sw_m, kit, cur_time_ms, frame_dims, other_sh_scr,
                mediapipe_present: bool = True,
                rhythm_session=None):
        """
        mediapipe_present — pass False when MediaPipe returned no landmarks
        for this side so the filter can extrapolate instead of freezing.
        """
        w, h = frame_dims
        wrist_px = (w_scr.x * w, w_scr.y * h)

        current_sw_px = math.hypot(
            sh_scr.x - other_sh_scr.x,
            sh_scr.y - other_sh_scr.y,
        ) * w
        if current_sw_px == 0:
            current_sw_px = 1

        # ── Kalman update / predict ───────────────────────────────────────
        if mediapipe_present:
            self._mediapipe_missing_frames = 0
            kx, ky, kz = self._kf.update(w_wrl.x, w_wrl.y, w_wrl.z)
        else:
            self._mediapipe_missing_frames += 1
            if self._mediapipe_missing_frames > self._MAX_MISSING_FRAMES:
                # Signal has been gone too long — reset so we don't drift.
                self._kf.reset()
                self._mediapipe_missing_frames = 0
                # Nothing useful to do this frame.
                return None, self._empty_debug(wrist_px)
            # Extrapolate: keep the filter running without a measurement.
            kx, ky, kz = self._kf.predict_only()

        curr_3d_coords = (kx, ky, kz)

        # ── 2-D screen-space motion (state machine only) ──────────────────
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

        # ── State machine ─────────────────────────────────────────────────
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
            world_y_delta = (
                curr_3d_coords[1] - self.prev_3d_coords[1]
                if self.prev_3d_coords is not None else 0.0
            )
            sw_dist = math.sqrt(
                (curr_3d_coords[0] - sh_wrl.x) ** 2 +
                (curr_3d_coords[1] - sh_wrl.y) ** 2 +
                (curr_3d_coords[2] - sh_wrl.z) ** 2
            )


            if (self.smooth_norm_speed > SPEED_THRESHOLD and
                    self.prev_3d_coords is not None and
                    downward_motion > 0 and
                    world_y_delta > self.WORLD_Y_STRIKE_THRESHOLD and
                    sw_dist > self.MIN_ARM_EXTENSION_M and
                    (cur_time_ms - self.last_hit_time) > COOLDOWN_MS):
        
                # ── Rhythm-stats hook: 2D trigger ──────────────────────────────
                if rhythm_session is not None:
                    rhythm_session.on_2d_trigger(
                        wrist_px    = wrist_px,
                        wrist_3d    = curr_3d_coords,
                        time_ms     = cur_time_ms,
                    )
                # ──────────────────────────────────────────────────────────────
        
                hit_detected = kit.check_line_intersection(
                    self.prev_3d_coords,
                    curr_3d_coords,
                    cur_time_ms / 1000.0,
                    self.smooth_norm_speed,
                    self.prev_wrist_px,
                    wrist_px,
                    self.label,
                )
        
                # ── Rhythm-stats hook: 3D result ───────────────────────────────
                if rhythm_session is not None:
                    rhythm_session.on_3d_result(
                        drum_name   = hit_detected,
                        wrist_px    = wrist_px,
                        wrist_3d    = curr_3d_coords,
                        time_ms     = cur_time_ms,
                    )
                # ──────────────────────────────────────────────────────────────
        
                #if hit_detected:  # [ DEBUG ]
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

        # ── Update per-frame memory ───────────────────────────────────────
        self.prev_wrist_px  = wrist_px
        self.prev_3d_coords = curr_3d_coords

        debug_info = {
            "pos_px":      (int(wrist_px[0]), int(wrist_px[1])),
            "sh_px":       (int(sh_scr.x * w), int(sh_scr.y * h)),
            "z":           kz,              # filtered Z shown in overlay
            "state":       "DOWN" if self.state == DOWN else "UP",
            "hit":         hit_detected,
            "debug_speed": self.smooth_norm_speed,
            "norm_3d":     curr_3d_coords,
        }

        return hit_detected, debug_info

    # ─────────────────────────────────────────────────────────────────────

    def _empty_debug(self, wrist_px):
        return {
            "pos_px":      (int(wrist_px[0]), int(wrist_px[1])),
            "sh_px":       (0, 0),
            "z":           0.0,
            "state":       "UP",
            "hit":         None,
            "debug_speed": 0.0,
            "norm_3d":     (0.0, 0.0, 0.0),
        }