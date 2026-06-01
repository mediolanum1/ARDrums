import cv2
import numpy as np
import math

# --- UPDATED IMPORT ---
from ARDrum_kit.processing.kalman_wrist import WristKalman

# Tune these HSV ranges per your tip color.
# Run the helper at bottom to find your values.
TIP_PROFILES = {
    "red":    ([0,  120, 100], [10, 255, 255],   # lower red hue
               [160, 120, 100], [180, 255, 255]), # upper red hue (wraps)
    "green":  ([40,  80,  80], [80, 255, 255], None, None),
    "blue":   ([100, 120,  80], [130, 255, 255], None, None),
    "orange": ([8,  150, 150], [20, 255, 255], None, None),
    "pink":   ([145, 80,  80], [165, 255, 255], None, None),
}

class ColorTipTracker:
    """
    Detects a colored drumstick tip in the frame, then computes the
    full 3-D position of the tip using:
      - wrist world-space coords from MediaPipe  (anchor point)
      - tip 2-D screen coords from color blob    (ray direction)
      - known stick length                       (depth constraint)

    Also runs a Kalman filter on the tip 3-D output to smooth jitter.
    """

    def __init__(self,
                 color: str,
                 stick_length_m: float,
                 focal_length: float,
                 frame_w: int,
                 frame_h: int,
                 min_blob_area: int = 40,
                 max_blob_area: int = 2000):

        assert color in TIP_PROFILES, f"Unknown color '{color}'. Choose from {list(TIP_PROFILES)}"
        profile = TIP_PROFILES[color]
        self._lo1  = np.array(profile[0], dtype=np.uint8)
        self._hi1  = np.array(profile[1], dtype=np.uint8)
        self._lo2  = np.array(profile[2], dtype=np.uint8) if profile[2] else None
        self._hi2  = np.array(profile[3], dtype=np.uint8) if profile[3] else None

        self.stick_length = stick_length_m
        self.focal_length = focal_length
        self.cx = frame_w / 2.0
        self.cy = frame_h / 2.0
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area

        # Kalman on the 3-D tip position
        self._kf = WristKalman(process_noise=8e-3, measurement_noise=6e-2)

        # Last known tip pixel (for drawing even when lost briefly)
        self.last_tip_px = None
        self._lost_frames = 0
        self._MAX_LOST = 6   # extrapolate for up to N frames then give up

    # ── Color blob detection ──────────────────────────────────────────────
    def _detect_tip_px(self, bgr_frame):
        """
        Returns (cx, cy) pixel of largest valid blob, or None.
        Applies a small blur first to reduce single-pixel noise.
        """
        hsv  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lo1, self._hi1)
        if self._lo2 is not None:
            mask |= cv2.inRange(hsv, self._lo2, self._hi2)

        # Morphological clean-up: kill noise, fill small holes
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if self.min_blob_area <= area <= self.max_blob_area:
                if area > best_area:
                    best_area = area
                    best = c

        if best is None:
            return None, mask

        M   = cv2.moments(best)
        if M["m00"] == 0:
            return None, mask
        cx  = M["m10"] / M["m00"]
        cy  = M["m01"] / M["m00"]
        
        # Optional: Print statement removed to avoid console spam during live play
        # print("[DEBUG] found a tip!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        return (cx, cy), mask

    # ── 3-D tip reconstruction ────────────────────────────────────────────
    def _solve_tip_3d(self, tip_px, wrist_world):
        """
        Intersects the camera ray through tip_px with a sphere of
        radius=stick_length centered at wrist_world.
        """
        wx, wy, wz = wrist_world

        # Unit ray direction through the tip pixel
        rx = (tip_px[0] - self.cx) / self.focal_length
        ry = (tip_px[1] - self.cy) / self.focal_length
        rz = 1.0
        mag = math.sqrt(rx*rx + ry*ry + rz*rz)
        rx /= mag; ry /= mag; rz /= mag

        # Quadratic  d² - 2d(r·W) + |W|² - L² = 0
        rW = rx*wx + ry*wy + rz*wz
        discriminant = rW*rW - (wx*wx + wy*wy + wz*wz) + self.stick_length**2

        if discriminant < 0:
            # Numerically no solution — clamp (float errors on fringe cases)
            discriminant = 0.0

        sqrt_disc = math.sqrt(discriminant)
        d1 = rW + sqrt_disc
        d2 = rW - sqrt_disc

        # Pick the positive root; if both positive take the one
        # whose z-component is closest to wrist depth (avoids far intersection)
        candidates = [d for d in (d1, d2) if d > 0.01]
        if not candidates:
            return None

        d = min(candidates, key=lambda d_: abs(d_*rz - wz))

        return (d*rx, d*ry, d*rz)

    # ── Main update ───────────────────────────────────────────────────────
    def update(self, bgr_frame, wrist_world):
        tip_px, dbg_mask = self._detect_tip_px(bgr_frame)

        if tip_px is not None:
            self._lost_frames = 0
            self.last_tip_px  = tip_px

            raw_3d = self._solve_tip_3d(tip_px, wrist_world)
            if raw_3d is not None:
                sx, sy, sz = self._kf.update(*raw_3d)
                return (sx, sy, sz), tip_px, dbg_mask

        else:
            self._lost_frames += 1
            if self._lost_frames <= self._MAX_LOST:
                # Extrapolate via Kalman predict
                sx, sy, sz = self._kf.predict_only()
                return (sx, sy, sz), self.last_tip_px, dbg_mask

        return None, None, dbg_mask

    # ── Debug drawing ─────────────────────────────────────────────────────
    def draw_debug(self, image, tip_px, tip_3d, color_bgr=(0, 255, 200)):
        if tip_px is None:
            return
        px = (int(tip_px[0]), int(tip_px[1]))
        cv2.circle(image, px, 10, color_bgr, 2)
        cv2.circle(image, px, 3,  color_bgr, -1)
        if tip_3d is not None:
            cv2.putText(image,
                        f"z={tip_3d[2]:.3f}",
                        (px[0] + 12, px[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1)


# ── HSV calibration helper ────────────────────────────────────────────────────
# Run this standalone to find HSV ranges for your tip color:
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    def nothing(x): pass
    cv2.namedWindow("Tune")
    for n, v in [("HL",0),("SL",0),("VL",0),("HH",179),("SH",255),("VH",255)]:
        cv2.createTrackbar(n, "Tune", v, 179 if n[0]=="H" else 255, nothing)
    while True:
        _, f = cap.read(); f = cv2.flip(f, 1)
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        lo = np.array([cv2.getTrackbarPos(n,"Tune") for n in ("HL","SL","VL")], np.uint8)
        hi = np.array([cv2.getTrackbarPos(n,"Tune") for n in ("HH","SH","VH")], np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        cv2.imshow("Tune", cv2.bitwise_and(f, f, mask=mask))
        if cv2.waitKey(1) == 27: break