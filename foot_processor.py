import math

UP   = 0  # kept for debug_info compatibility with the rest of the app
DOWN = 1

FOOT_HIT_DY_THRESHOLD = 0.030   # normalised downward dy in a single frame to count as a hit
                                  # (norm_dy = pixel_dy / hip_width_px)
                                  # raise if getting false positives from walking/fidgeting
                                  # lower if pedal presses aren't registering
FOOT_COOLDOWN_MS      = 200       # minimum ms between consecutive bass hits


class GestureFootProcessor:
    """
    Simplified bass-drum pedal detector.

    A hit is registered whenever the ankle moves downward by more than
    FOOT_HIT_DY_THRESHOLD (normalised by hip width) in a single frame,
    subject to a per-hit cooldown.  No state machine, no world-Z check.
    """

    def __init__(self, label: str):
        self.label             = label   # "L" or "R"
        self.last_hit_time     = 0       # ms
        self.prev_ankle_px     = None
        self._last_state       = UP      # only used so debug overlay has something to show

    def process(self,
                ankle_scr,
                ankle_wrl,        # kept in signature for API compatibility; unused
                hip_scr,
                other_hip_scr,
                kit,
                cur_time_ms: int,
                frame_dims:  tuple,
                mediapipe_present: bool = True):

        w, h = frame_dims
        ankle_px = (ankle_scr.x * w, ankle_scr.y * h)

        if not mediapipe_present:
            self.prev_ankle_px = None
            return None, self._empty_debug(ankle_px)

        # Normalise by hip-width pixels so threshold is distance-independent
        hip_w_px = math.hypot(
            (hip_scr.x - other_hip_scr.x) * w,
            (hip_scr.y - other_hip_scr.y) * h,
        )
        hip_w_px = max(hip_w_px, 1.0)

        hit_detected = None

        if self.prev_ankle_px is not None:
            norm_dy = (ankle_px[1] - self.prev_ankle_px[1]) / hip_w_px

            if (norm_dy > FOOT_HIT_DY_THRESHOLD and
                    (cur_time_ms - self.last_hit_time) > FOOT_COOLDOWN_MS):

                hit_detected = kit.trigger_bass_drum(
                    cur_time          = cur_time_ms / 1000.0,
                    smooth_norm_speed = norm_dy,   # used for volume scaling
                    hand_id           = self.label,
                )
                if hit_detected:
                    self.last_hit_time = cur_time_ms

            # Drive the state purely for the visual debug ring
            self._last_state = DOWN if norm_dy > FOOT_HIT_DY_THRESHOLD * 0.4 else UP

        self.prev_ankle_px = ankle_px

        return hit_detected, {
            "pos_px":      (int(ankle_px[0]), int(ankle_px[1])),
            "state":       "DOWN" if self._last_state == DOWN else "UP",
            "hit":         hit_detected,
            "debug_speed": 0.0,
            "norm_3d":     (0.0, 0.0, 0.0),
        }

    def _empty_debug(self, ankle_px):
        return {
            "pos_px":      (int(ankle_px[0]), int(ankle_px[1])),
            "state":       "UP",
            "hit":         None,
            "debug_speed": 0.0,
            "norm_3d":     (0.0, 0.0, 0.0),
        }