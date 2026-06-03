import math

UP = 0
DOWN = 1

FOOT_SPEED_THRESHOLD           = 0.01 
FOOT_HIT_DY_THRESHOLD         = 0.05    # normalised downward dy in a single frame to count as a hit
FOOT_COOLDOWN_MS               = 200     # minimum ms between consecutive bass hits
FOOT_STATE_CHANGE_FRAME_THRESH = 1
FOOT_MIN_UPWARD_MOTION         = -0.01
FOOT_STALL_RESET_FRAME_THRESH  = 12
FOOT_STALL_SPEED_THRESHOLD     = 0.008


class GestureFootProcessor:
    """
    Bass-drum pedal detector with a simple UP/DOWN state machine.

    A hit is emitted once when the ankle transitions into a downward press,
    then the processor stays in DOWN until the foot lifts or stalls.
    """

    def __init__(self, label: str):
        self.label              = label
        self.state              = UP
        self.state_change_frame = 0
        self.last_hit_time      = 0
        self.prev_ankle_px      = None
        self.smooth_norm_speed  = 0.0

    def process(self,
                ankle_scr,
                ankle_wrl,        # kept in signature for API compatibility; unused
                hip_scr,
                other_hip_scr,
                kit,
                cur_time_ms: int,
                frame_dims: tuple,
                mediapipe_present: bool = True):

        w, h = frame_dims
        ankle_px = (ankle_scr.x * w, ankle_scr.y * h)

        if not mediapipe_present:
            self.prev_ankle_px = None
            self.state = UP
            self.state_change_frame = 0
            return None, self._empty_debug(ankle_px)

        hip_w_px = math.hypot(
            (hip_scr.x - other_hip_scr.x) * w,
            (hip_scr.y - other_hip_scr.y) * h,
        )
        hip_w_px = max(hip_w_px, 1.0)

        norm_dx = 0.0
        norm_dy = 0.0
        raw_norm_speed = 0.0

        if self.prev_ankle_px is not None:
            dx = ankle_px[0] - self.prev_ankle_px[0]
            dy = ankle_px[1] - self.prev_ankle_px[1]
            norm_dx = dx / hip_w_px
            norm_dy = dy / hip_w_px
            raw_norm_speed = math.hypot(norm_dx, norm_dy)

        self.smooth_norm_speed = (self.smooth_norm_speed * 0.3) + (raw_norm_speed * 0.7)

        downward_motion = norm_dy
        upward_motion = norm_dy

        hit_detected = None

        if self.state == UP:
            if (self.smooth_norm_speed > FOOT_SPEED_THRESHOLD and
                    downward_motion > FOOT_MIN_DOWNWARD_MOTION):
                self.state_change_frame += 1
                if self.state_change_frame > FOOT_STATE_CHANGE_FRAME_THRESH:
                    self.state = DOWN
                    self.state_change_frame = 0
                    if (self.smooth_norm_speed > FOOT_SPEED_THRESHOLD and
                            downward_motion > FOOT_HIT_DY_THRESHOLD and
                            cur_time_ms - self.last_hit_time > FOOT_COOLDOWN_MS):
                        hit_detected = kit.trigger_bass_drum(
                            cur_time          = cur_time_ms / 1000.0,
                            smooth_norm_speed = self.smooth_norm_speed,
                            hand_id           = self.label,
                        )
                        if hit_detected:
                            self.last_hit_time = cur_time_ms
            else:
                self.state_change_frame = 0

        elif self.state == DOWN:
            if (self.smooth_norm_speed > FOOT_SPEED_THRESHOLD and
                    downward_motion > FOOT_HIT_DY_THRESHOLD and
                    cur_time_ms - self.last_hit_time > FOOT_COOLDOWN_MS):
                hit_detected = kit.trigger_bass_drum(
                    cur_time          = cur_time_ms / 1000.0,
                    smooth_norm_speed = self.smooth_norm_speed,
                    hand_id           = self.label,
                )
                if hit_detected:
                    self.last_hit_time = cur_time_ms

            if upward_motion < FOOT_MIN_UPWARD_MOTION:
                self.state_change_frame += 1
                if self.state_change_frame > FOOT_STATE_CHANGE_FRAME_THRESH:
                    self.state = UP
                    self.state_change_frame = 0
            elif self.smooth_norm_speed < FOOT_STALL_SPEED_THRESHOLD:
                self.state_change_frame += 1
                if self.state_change_frame > FOOT_STALL_RESET_FRAME_THRESH:
                    self.state = UP
                    self.state_change_frame = 0
            else:
                self.state_change_frame = 0

        self.prev_ankle_px = ankle_px

        return hit_detected, {
            "pos_px":      (int(ankle_px[0]), int(ankle_px[1])),
            "state":       "DOWN" if self.state == DOWN else "UP",
            "hit":         hit_detected,
            "debug_speed": self.smooth_norm_speed,
            "norm_3d":     (ankle_wrl.x, ankle_wrl.y, ankle_wrl.z),
        }

    def _empty_debug(self, ankle_px):
        return {
            "pos_px":      (int(ankle_px[0]), int(ankle_px[1])),
            "state":       "UP",
            "hit":         None,
            "debug_speed": 0.0,
            "norm_3d":     (0.0, 0.0, 0.0),
        }
