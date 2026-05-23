import threading
import queue
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
import time
import math
import os
import pygame

from processors import GestureWristProcessor
from stats_collector import StatsCollector
import numpy as np
from drum import VirtualDrumKit


APP_H     = 820
RIGHT_W   = 640
POV_H     = 540
WIN_NAME  = "AR Drum Kit"

_POV_REN_W = 800
_POV_REN_H = 640

class ARDrumApp:
    def __init__(self):
        self.frame_queue  = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running      = True

        self.cap          = cv2.VideoCapture(1)
        self.frame_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.focal_length = (self.frame_width / 2) / math.tan(math.radians(65.0) / 2)
        self.kit          = VirtualDrumKit()

        self.is_calibrated         = False
        self.fixed_sw_m            = 1.0
        self.cached_drum_positions = None
        self.static_drum_positions = None
        
        # Scale for converting pure meters to 2D pixels for the web cam overlay
        self.metric_to_px_scale = 1.0 
        self._current_sw_px     = 120.0

        self.show_drums        = True
        self.show_coords       = False
        self.show_occlusion    = False
        self.show_hit_messages = False
        self.show_flow          = False
        self.show_drum_names   = True
        self.show_left_state   = False
        self.show_pov          = True

        self.optical_flow_enabled = True
        self.prev_gray           = None
        self.prev_wrist_px       = {"left": None, "right": None}

        self.stick_mode   = False
        self.stick_length = 0
        self._stick_ext_l = (0.0, 0.0, 0.0)
        self._stick_ext_r = (0.0, 0.0, 0.0)

        self._last_stick_dir_l = (0.0, 1.0)
        self._last_stick_dir_r = (0.0, 1.0)
        self._STICK_MIN_ARM_PX = 22
        self._STICK_BLEND = 0.55

        self.last_l_hit_time = 0
        self.last_r_hit_time = 0

        self.program_start_time = time.time()
        self.COUNTDOWN_SECONDS  = 5
        self.stats = StatsCollector()

        self.left_arm  = GestureWristProcessor("Left ")
        self.right_arm = GestureWristProcessor("Right ")

    def camera_thread(self):
        while self.running:
            success, image = self.cap.read()
            if not success: continue
            image = cv2.flip(image, 1)
            filtered_image = cv2.Laplacian(image, cv2.CV_16S, ksize=3)
            image = cv2.subtract(image,filtered_image)
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                continue
            self.frame_queue.put(image)

    def ai_thread(self):
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="./pose_landmarker_models/pose_landmarker_full.task"),
            running_mode=vision.RunningMode.VIDEO,
        )

        with PoseLandmarker.create_from_options(options) as pose_landmarker:
            while self.running:
                image = self.frame_queue.get()
                rgb   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ts_ms = int(time.time() * 1000)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                result = pose_landmarker.detect_for_video(mp_img, ts_ms)

                if not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.result_queue.put((image, result, time.time()))

    def main_render_loop(self):
        while self.running:
            try:
                image, result, cur_time = self.result_queue.get(timeout=0.1)
            except queue.Empty:
                if 'image' in locals():
                    self._show_combined(image, None, None, None, time.time())
                continue

            if result.pose_landmarks and result.pose_world_landmarks:
                s_lm = result.pose_landmarks[0]
                w_lm = result.pose_world_landmarks[0]

                if not self.is_calibrated:
                    elapsed = cur_time - self.program_start_time
                    if elapsed < self.COUNTDOWN_SECONDS:
                        cv2.putText(
                            image, f"READY IN: {int(self.COUNTDOWN_SECONDS - elapsed)}",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        )
                    else:
                        self._calibrate(s_lm, w_lm)

                if self.is_calibrated:
                    self._update_drum_positions(s_lm)
                    cur_time_ms = int(time.time() * 1000)
                    dims = (self.frame_width, self.frame_height)

                    if self.stick_mode:
                        self._stick_ext_l = self._compute_stick_ext(w_lm[13], w_lm[15])
                        self._stick_ext_r = self._compute_stick_ext(w_lm[14], w_lm[16])
                    else:
                        self._stick_ext_l = (0.0, 0.0, 0.0)
                        self._stick_ext_r = (0.0, 0.0, 0.0)

                    left_wrist_px = (s_lm[15].x * self.frame_width, s_lm[15].y * self.frame_height)
                    right_wrist_px = (s_lm[16].x * self.frame_width, s_lm[16].y * self.frame_height)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    if self.optical_flow_enabled:
                        left_flow, right_flow = self._compute_wrist_optical_flow(gray, left_wrist_px, right_wrist_px)
                    else:
                        left_flow, right_flow = None, None

                    self.kit.active_stick_ext = self._stick_ext_l
                    hit_l, dbg_l = self.left_arm.process(
                        s_lm[15], w_lm[15], s_lm[11], w_lm[11], s_lm[13],
                        1.0, self.kit, cur_time_ms, dims, s_lm[12], left_flow,
                    )

                    self.kit.active_stick_ext = self._stick_ext_r
                    hit_r, dbg_r = self.right_arm.process(
                        s_lm[16], w_lm[16], s_lm[12], w_lm[12], s_lm[14],
                        1.0, self.kit, cur_time_ms, dims, s_lm[11], right_flow,
                    )
                    self.kit.active_stick_ext = (0.0, 0.0, 0.0)

                    self.prev_gray = gray
                    self.prev_wrist_px["left"] = left_wrist_px
                    self.prev_wrist_px["right"] = right_wrist_px

                    if hit_l: self.last_l_hit_time = cur_time
                    if hit_r: self.last_r_hit_time = cur_time

                    if self.show_hit_messages:
                        if cur_time - self.last_l_hit_time < 0.5:
                            cv2.putText(image, "RIGHT HAND HIT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        if cur_time - self.last_r_hit_time < 0.5:
                            cv2.putText(image, "LEFT HAND HIT!", (self.frame_width - 350, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                    self._draw_arm_debug(image, dbg_l, (255, 0, 0))
                    self._draw_arm_debug(image, dbg_r, (0, 0, 255))

                    if self.stick_mode:
                        self._draw_stick_ar(image, s_lm[13], s_lm[15], (255, 200, 50), side='l')
                        self._draw_stick_ar(image, s_lm[14], s_lm[16], (50, 220, 255), side='r')
                    else:
                        self._draw_hand_ext(image, s_lm[13], s_lm[15], (255, 0, 0))
                        self._draw_hand_ext(image, s_lm[14], s_lm[16], (0, 0, 255))

                    if self.show_drums and self.static_drum_positions:
                        self._draw_drums(image, cur_time)

                    if self.show_flow:
                        self._draw_optical_flow(image, left_wrist_px, left_flow, (0, 255, 255))
                        self._draw_optical_flow(image, right_wrist_px, right_flow, (255, 255, 0))

                    pov_canvas = self._render_pov_canvas(dbg_l, dbg_r, cur_time, w_lm) if self.show_pov else None
                    self._show_combined(image, pov_canvas, dbg_l, dbg_r, cur_time)
                else:
                    self._show_combined(image, None, None, None, cur_time)
            else:
                self._show_combined(image, None, None, None, cur_time)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.running = False
            elif key == ord("d"): self.show_drums        = not self.show_drums
            elif key == ord("c"): self.show_coords       = not self.show_coords
            elif key == ord("h"): self.show_occlusion    = not self.show_occlusion
            elif key == ord("j"): self.show_hit_messages = not self.show_hit_messages
            elif key == ord("n"): self.show_drum_names   = not self.show_drum_names
            elif key == ord("o"): self.show_flow        = not self.show_flow
            elif key == ord("y"): self.show_left_state   = not self.show_left_state
            elif key == ord("p"): self.show_pov          = not self.show_pov
            elif key == ord("s"):
                self.stick_mode     = not self.stick_mode
                self.kit.use_sticks = self.stick_mode

    def _show_combined(self, cam_img, pov_canvas, dbg_l, dbg_r, cur_time):
        cam_w   = int(self.frame_width * (APP_H / self.frame_height))
        total_w = cam_w + RIGHT_W
        ctrl_h  = APP_H - POV_H

        combined = np.zeros((APP_H, total_w, 3), dtype=np.uint8)

        cam = cv2.resize(cam_img, (cam_w, APP_H))
        combined[:, :cam_w] = cam
        cv2.line(combined, (cam_w, 0), (cam_w, APP_H), (40, 40, 40), 2)

        if self.show_pov and pov_canvas is not None:
            pov_resized = cv2.resize(pov_canvas, (RIGHT_W, POV_H))
        else:
            pov_resized = np.full((POV_H, RIGHT_W, 3), (12, 14, 22), dtype=np.uint8)
            cv2.putText(pov_resized, "POV hidden  [P] to show", (RIGHT_W // 2 - 130, POV_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 80), 1)

        combined[:POV_H, cam_w:] = pov_resized
        cv2.line(combined, (cam_w, POV_H), (total_w, POV_H), (40, 40, 40), 2)

        ctrl_panel = self._build_controls_panel(RIGHT_W, ctrl_h, dbg_l, dbg_r, cur_time)
        combined[POV_H:, cam_w:] = ctrl_panel

        cv2.imshow(WIN_NAME, combined)

    def _build_controls_panel(self, w, h, dbg_l, dbg_r, cur_time):
        panel = np.full((h, w, 3), (14, 16, 24), dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (w, 32), (24, 28, 42), -1)
        cv2.putText(panel, "CONTROLS", (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 210, 210), 1)

        keys = [
            ("[D]",   "Toggle drums",   self.show_drums),
            ("[N]",   "Drum names",     self.show_drum_names),
            ("[P]",   "POV window",     self.show_pov),
            ("[S]",   "Stick mode",     self.stick_mode),
            ("[C]",   "Coords overlay", self.show_coords),
            ("[H]",   "Occlusion ring", self.show_occlusion),
            ("[J]",   "Hit messages",   self.show_hit_messages),
            ("[O]",   "Flow debug",     self.show_flow),
            ("[Y]",   "Left-arm state", self.show_left_state),
            ("[ESC]", "Quit",           None),
        ]

        row_h = max(26, (h - 50) // len(keys))
        for i, (key, label, state) in enumerate(keys):
            y  = 44 + i * row_h
            kw = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
            cv2.rectangle(panel, (10, y - 14), (10 + kw + 10, y + 6), (34, 38, 60), -1)
            cv2.rectangle(panel, (10, y - 14), (10 + kw + 10, y + 6), (60, 65, 100), 1)
            cv2.putText(panel, key,   (15, y),           cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 255), 1)
            cv2.putText(panel, label, (10 + kw + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 180), 1)

            if state is not None:
                on_col, off_col = (0, 220, 110), (80, 80, 100)
                dot_col = on_col if state else off_col
                dot_x   = w - 22
                cv2.circle(panel, (dot_x, y - 4), 6, dot_col, -1)
                lbl = "ON" if state else "off"
                cv2.putText(panel, lbl, (dot_x - 26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, dot_col, 1)

        base_y = h - 18
        if dbg_l and (cur_time - self.last_l_hit_time) < 0.4:
            cv2.putText(panel, "LEFT  HIT!", (14, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)
        if dbg_r and (cur_time - self.last_r_hit_time) < 0.4:
            cv2.putText(panel, "RIGHT HIT!", (w // 2 + 10, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)

        return panel

    def _calibrate(self, s_lm, w_lm):
        l_sh_w, r_sh_w = w_lm[11], w_lm[12]
        l_sh_s, r_sh_s = s_lm[11], s_lm[12]
        if l_sh_s.visibility <= 0.5 or r_sh_s.visibility <= 0.5:
            return

        self.fixed_sw_m = math.sqrt((l_sh_w.x - r_sh_w.x)**2 + (l_sh_w.y - r_sh_w.y)**2 + (l_sh_w.z - r_sh_w.z)**2)
        fixed_sw_px = math.hypot((l_sh_s.x - r_sh_s.x) * self.frame_width, (l_sh_s.y - r_sh_s.y) * self.frame_height)

        self.metric_to_px_scale = fixed_sw_px / self.fixed_sw_m if self.fixed_sw_m > 0 else 1.0

        print(f"[CAL] sw={self.fixed_sw_m:.3f} m")
        self.is_calibrated = True

    def _update_drum_positions(self, s_lm):
        l_sh_s, r_sh_s = s_lm[11], s_lm[12]
        self._current_sw_px = max(1, math.hypot((l_sh_s.x - r_sh_s.x) * self.frame_width, (l_sh_s.y - r_sh_s.y) * self.frame_height))
        
        # Maintain pixel scaling factor across varying distances
        self.metric_to_px_scale = self._current_sw_px / self.fixed_sw_m if self.fixed_sw_m > 0 else 1.0

        if self.static_drum_positions is not None:
            self.cached_drum_positions = self.static_drum_positions
            return

        anchor_x = int(((s_lm[23].x + s_lm[24].x) / 2) * self.frame_width)
        anchor_y = int(((s_lm[23].y + s_lm[24].y) / 2) * self.frame_height)

        positions = {}
        for name, props in self.kit.drums.items():
            cx_m, cy_m, _ = props["center"]
            rx_m, ry_m, _ = props["radii"]
            
            # Draw exactly the hitbox sizes using the real metric_to_px scale
            positions[name] = {
                "cx": int(anchor_x + cx_m * self.metric_to_px_scale),
                "cy": int(anchor_y + cy_m * self.metric_to_px_scale),
                "rx": int(rx_m * self.metric_to_px_scale),
                "ry": int(ry_m * self.metric_to_px_scale),
            }
            
        self.cached_drum_positions  = positions
        self.static_drum_positions  = positions

    def _compute_stick_ext(self, elbow_w, wrist_w):
        dx, dy, dz = wrist_w.x - elbow_w.x, wrist_w.y - elbow_w.y, wrist_w.z - elbow_w.z
        mag = math.sqrt(dx**2 + dy**2 + dz**2)
        if mag < 1e-6: return (0.0, 0.0, 0.0)
        return (dx / mag * self.stick_length, dy / mag * self.stick_length, dz / mag * self.stick_length)

    def _draw_stick_ar(self, image, elbow_s, wrist_s, color, side='r'):
        W, H = self.frame_width, self.frame_height
        ew = (int(elbow_s.x * W), int(elbow_s.y * H))
        wr = (int(wrist_s.x * W), int(wrist_s.y * H))
        dx = wr[0] - ew[0]
        dy = wr[1] - ew[1]
        arm_len_px = math.hypot(dx, dy)

        cached = self._last_stick_dir_l if side == 'l' else self._last_stick_dir_r

        if arm_len_px >= self._STICK_MIN_ARM_PX:
            ndx, ndy = dx / arm_len_px, dy / arm_len_px
            bx = cached[0] * (1 - self._STICK_BLEND) + ndx * self._STICK_BLEND
            by = cached[1] * (1 - self._STICK_BLEND) + ndy * self._STICK_BLEND
            mag = math.hypot(bx, by)
            if mag > 1e-6: bx, by = bx / mag, by / mag
            new_dir = (bx, by)
        else:
            bx = cached[0] * 0.97 + 0.0 * 0.03
            by = cached[1] * 0.97 + 1.0 * 0.03
            mag = math.hypot(bx, by)
            new_dir = (bx / mag, by / mag) if mag > 1e-6 else cached

        if side == 'l': self._last_stick_dir_l = new_dir
        else: self._last_stick_dir_r = new_dir

        ndx, ndy = new_dir
        stick_px_len = self.stick_length * self.metric_to_px_scale
        tip = (int(wr[0] + ndx * stick_px_len), int(wr[1] + ndy * stick_px_len))
        cv2.line(image, wr, tip, color, 4)
        cv2.circle(image, tip, 7, (255, 240, 80), -1)

    def _draw_hand_ext(self, image, elbow_s, wrist_s, color):
        W, H = self.frame_width, self.frame_height
        ew = (int(elbow_s.x * W), int(elbow_s.y * H))
        wr = (int(wrist_s.x * W), int(wrist_s.y * H))
        dx = wr[0] - ew[0]
        dy = wr[1] - ew[1]
        arm_len_px = math.hypot(dx, dy)
        if arm_len_px < 1: return
        hand_px_len = self.metric_to_px_scale * 0.15
        tip = (int(wr[0] + (dx / arm_len_px) * hand_px_len), int(wr[1] + (dy / arm_len_px) * hand_px_len))
        cv2.line(image, wr, tip, color, 3)
        cv2.circle(image, tip, 5, color, -1)

    def _draw_optical_flow(self, image, wrist_px, flow, color):
        if flow is None:
            return
        start = (int(wrist_px[0]), int(wrist_px[1]))
        end = (int(wrist_px[0] + flow[0] * 3), int(wrist_px[1] + flow[1] * 3))
        cv2.arrowedLine(image, start, end, color, 2, tipLength=0.3)

    def _compute_wrist_optical_flow(self, gray, left_px, right_px):
        if self.prev_gray is None:
            return None, None

        prev_left = self.prev_wrist_px.get("left")
        prev_right = self.prev_wrist_px.get("right")
        if prev_left is None or prev_right is None:
            return None, None

        prev_pts = np.array([prev_left, prev_right], dtype=np.float32).reshape(-1, 1, 2)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            prev_pts,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

        left_flow = None
        right_flow = None
        if curr_pts is not None and status is not None:
            if status.shape[0] > 0 and status[0][0] == 1:
                left_flow = (float(curr_pts[0, 0, 0] - prev_left[0]), float(curr_pts[0, 0, 1] - prev_left[1]))
            if status.shape[0] > 1 and status[1][0] == 1:
                right_flow = (float(curr_pts[1, 0, 0] - prev_right[0]), float(curr_pts[1, 0, 1] - prev_right[1]))

        if left_flow is None and left_px is not None:
            left_flow = (left_px[0] - prev_left[0], left_px[1] - prev_left[1])
        if right_flow is None and right_px is not None:
            right_flow = (right_px[0] - prev_right[0], right_px[1] - prev_right[1])

        return left_flow, right_flow

    def _draw_arm_debug(self, image, dbg, color):
        px = dbg["pos_px"]
        cv2.circle(image, px, 15, (0, 255, 0) if dbg["hit"] else color, -1)
        if self.show_coords:
            cv2.putText(image, f"STATE:{dbg['state']} Z:{dbg['z']:.2f}", (px[0] - 40, px[1] - 40), 0, 1.2, (0, 0, 255), 2)

    def _draw_drums(self, image, cur_time):
        overlay = image.copy()
        for name, pos in self.static_drum_positions.items():
            is_hit = (cur_time - self.kit.last_hit_time[name]) < 0.15
            color  = (0, 255, 0) if is_hit else self.kit.drums[name]["color_idle"]
            cv2.ellipse(overlay, (pos["cx"], pos["cy"]), (max(pos["rx"], 4), max(pos["ry"], 2)), 0, 0, 360, color, -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        if self.show_drum_names:
            for name, pos in self.static_drum_positions.items():
                label = name.upper()
                ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                tx, ty = pos["cx"] - ts[0] // 2, pos["cy"] + ts[1] // 2
                cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def _render_pov_canvas(self, dbg_l, dbg_r, cur_time, w_lm=None):
        canvas = np.full((_POV_REN_H, _POV_REN_W, 3), (8, 10, 18), dtype=np.uint8)

        SCALE = 140
        cx_c = _POV_REN_W // 2
        cy_c = _POV_REN_H // 2 - 20

        def project(x, y, z):
            depth = z
            dist  = -depth if depth < 0 else depth
            if dist < 0.1: dist = 0.1
            px = int(cx_c + (x / dist) * SCALE)
            py = int(cy_c + (y / dist) * SCALE)
            return px, py, depth, dist

        rq = []

        # ── Grid ──────────────────────────────────────────────────────────
        GRID_Y = 0.08
        XS = [-1.1, -0.7, -0.28, 0, 0.28, 0.7, 1.1]
        ZS = [-0.35, -0.55, -0.75, -0.95, -1.15]
        for gx in XS:
            p1, p2 = project(gx, GRID_Y, ZS[0]), project(gx, GRID_Y, ZS[-1])
            rq.append({"t": "line", "depth": (p1[2]+p2[2])/2, "p1": p1[:2], "p2": p2[:2], "color": (30, 42, 58), "w": 1})
        for gz in ZS:
            p1, p2 = project(XS[0], GRID_Y, gz), project(XS[-1], GRID_Y, gz)
            rq.append({"t": "line", "depth": (p1[2]+p2[2])/2, "p1": p1[:2], "p2": p2[:2], "color": (30, 42, 58), "w": 1})

        # ── Drums (Accurate True 1:1 Metric Scale) ────────────────────────
        for name, props in self.kit.drums.items():
            cx, cy, cz = props["center"]
            is_hit = (cur_time - self.kit.last_hit_time[name]) < 0.20
            col    = (0, 240, 60) if is_hit else props["color_idle"]
            ppx, ppy, depth, dist = project(cx, cy, cz)

            rx_m, ry_m, rz_m = props["radii"]
            rx    = max(int((rx_m * SCALE) / dist), 4)
            ry    = max(int((ry_m * SCALE) / dist), 2)
            thick = max(int((rz_m * 2 * SCALE) / dist), 2)

            rq.append({"t": "drum", "depth": depth, "name": name,
                       "px": ppx, "py": ppy, "rx": rx, "ry": ry, "thick": thick, "col": col})

        # ── Arms ──────────────────────────────────────────────────────────
        sw = self.fixed_sw_m
        if w_lm and sw > 0:
            arm_defs = [
                (w_lm[11], w_lm[13], w_lm[15], dbg_l, (100, 220, 255)),
                (w_lm[12], w_lm[14], w_lm[16], dbg_r, ( 80,  80, 255)),
            ]
            for sh_w, el_w, wr_w, dbg, arm_col in arm_defs:
                if sh_w.visibility < 0.3 or el_w.visibility < 0.3 or wr_w.visibility < 0.3:
                    continue

                sh3 = (sh_w.x, sh_w.y, sh_w.z)
                el3 = (el_w.x, el_w.y, el_w.z)
                wr3 = (wr_w.x, wr_w.y, wr_w.z)

                sh_p, el_p, wr_p = project(*sh3), project(*el3), project(*wr3)
                line_w = max(2, int(8 / el_p[3]))
                rq.append({"t": "line", "depth": (sh_p[2]+el_p[2])/2, "p1": sh_p[:2], "p2": el_p[:2], "color": arm_col, "w": line_w})
                rq.append({"t": "line", "depth": (el_p[2]+wr_p[2])/2, "p1": el_p[:2], "p2": wr_p[:2], "color": arm_col, "w": line_w})

                rad_sh = max(3, int(10 / sh_p[3]))
                rad_el = max(2, int( 8 / el_p[3]))
                rq.append({"t": "dot", "depth": sh_p[2], "px": sh_p[0], "py": sh_p[1], "col": arm_col, "r": rad_sh})
                rq.append({"t": "dot", "depth": el_p[2], "px": el_p[0], "py": el_p[1], "col": arm_col, "r": rad_el})

                fw_x, fw_y, fw_z = wr3[0] - el3[0], wr3[1] - el3[1], wr3[2] - el3[2]
                fw_mag = math.sqrt(fw_x**2 + fw_y**2 + fw_z**2)

                if self.stick_mode and fw_mag > 1e-3:
                    ext_len = self.stick_length
                    tip3 = (wr3[0] + (fw_x/fw_mag)*ext_len, wr3[1] + (fw_y/fw_mag)*ext_len, wr3[2] + (fw_z/fw_mag)*ext_len)
                    tp_p = project(*tip3)
                    stick_w = max(1, int(6 / tp_p[3]))
                    tip_r   = max(2, int(10 / tp_p[3]))
                    rq.append({"t": "line", "depth": (wr_p[2]+tp_p[2])/2, "p1": wr_p[:2], "p2": tp_p[:2], "color": (255, 220, 50), "w": stick_w})
                    rq.append({"t": "dot", "depth": tp_p[2], "px": tp_p[0], "py": tp_p[1], "col": (255, 220, 50), "r": tip_r})
                elif not self.stick_mode and fw_mag > 1e-3:
                    hand_len = 0.15
                    htip3 = (wr3[0] + (fw_x/fw_mag)*hand_len, wr3[1] + (fw_y/fw_mag)*hand_len, wr3[2] + (fw_z/fw_mag)*hand_len)
                    htp_p = project(*htip3)
                    hand_w = max(1, int(5 / htp_p[3]))
                    hand_r = max(2, int(6 / htp_p[3]))
                    rq.append({"t": "line", "depth": (wr_p[2]+htp_p[2])/2, "p1": wr_p[:2], "p2": htp_p[:2], "color": arm_col, "w": hand_w})
                    rq.append({"t": "dot", "depth": htp_p[2], "px": htp_p[0], "py": htp_p[1], "col": arm_col, "r": hand_r})

                is_hit_wrist = dbg.get("hit", False)
                wrist_col    = (0, 255, 60) if is_hit_wrist else arm_col
                rad_wr = max(4, int(14 / wr_p[3]))
                rq.append({"t": "hand", "depth": wr_p[2], "px": wr_p[0], "py": wr_p[1], "col": wrist_col, "state": dbg.get("state", ""), "r": rad_wr})

        rq.sort(key=lambda i: i["depth"])

        for item in rq:
            t = item["t"]
            if t == "line": cv2.line(canvas, item["p1"], item["p2"], item["color"], item["w"])
            elif t == "dot": cv2.circle(canvas, (item["px"], item["py"]), item["r"], item["col"], -1)
            elif t == "drum":
                ppx, ppy = item["px"], item["py"]
                rx, ry, thick = item["rx"], item["ry"], item["thick"]
                c = item["col"]
                shade = (int(c[0]*0.45), int(c[1]*0.45), int(c[2]*0.45))
                cv2.ellipse(canvas, (ppx, ppy+thick), (rx, ry), 0, 0, 360, shade, -1)
                pts = np.array([(ppx-rx, ppy), (ppx+rx, ppy), (ppx+rx, ppy+thick), (ppx-rx, ppy+thick)], dtype=np.int32)
                cv2.fillPoly(canvas, [pts], shade)
                cv2.ellipse(canvas, (ppx, ppy), (rx, ry), 0, 0, 360, c, -1)
                cv2.ellipse(canvas, (ppx, ppy), (rx, ry), 0, 0, 360, (170, 170, 170), 1)
                lbl = item["name"][:3].upper()
                ts  = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
                cv2.putText(canvas, lbl, (ppx - ts[0]//2, ppy + ts[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
                cv2.putText(canvas, lbl, (ppx - ts[0]//2, ppy + ts[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            elif t == "hand":
                ppx, ppy, r, c = item["px"], item["py"], item["r"], item["col"]
                if item["state"] == "DOWN": cv2.circle(canvas, (ppx, ppy), r + 8, c, 2)
                cv2.circle(canvas, (ppx, ppy), r, c, -1)
                cv2.circle(canvas, (ppx, ppy), r, (255, 255, 255), 1)

        cv2.putText(canvas, "TRUE 1ST PERSON POV", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 210), 1)
        return canvas

    def start(self):
        threading.Thread(target=self.camera_thread, daemon=True).start()
        threading.Thread(target=self.ai_thread,     daemon=True).start()
        self.main_render_loop()
        self.cap.release()
        cv2.destroyAllWindows()
        self.kit.cleanup()
        self.stats.save()

if __name__ == "__main__":
    ARDrumApp().start()