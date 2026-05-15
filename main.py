
import threading
import queue
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
import time
import math
from drum import VirtualDrumKit
from processors import GestureWristProcessor

# Using Mediapipe Tasks (vision/python) API exclusively.

class ARDrumApp:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=2)  
        self.result_queue = queue.Queue(maxsize=2) 
        self.running = True
        
        self.cap = cv2.VideoCapture(1)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.focal_length = (self.frame_width / 2) / math.tan(math.radians(65.0) / 2)
        self.kit = VirtualDrumKit()
        
        # --- State Flags ---
        self.is_calibrated = False
        self.show_drums = True 
        self.show_coords = False    
        self.show_occlusion = False 
        self.show_hit_messages = False  # New Flag for 'j'
        self.show_drum_names = True     # <--- NEW: Toggle for 'n'
        self.show_left_state = False    # <--- NEW: Toggle for 'y'
        
        # Timestamps to keep hit text on screen for a moment
        self.last_l_hit_time = 0
        self.last_r_hit_time = 0
        
        self.fixed_sw_m = 1.0   
        self.cached_drum_positions = None 
        
        self.program_start_time = time.time()
        self.COUNTDOWN_SECONDS = 5

        self.left_arm = GestureWristProcessor("Left ")
        self.right_arm = GestureWristProcessor("Right ")

    # ... [camera_thread and ai_thread remain the same] ...
    def camera_thread(self):
        while self.running:
            success, image = self.cap.read()
            if not success: continue
            image = cv2.flip(image, 1)
            if not self.frame_queue.empty():
                try: self.frame_queue.get_nowait()
                except queue.Empty: pass
            self.frame_queue.put(image)

    def ai_thread(self):
        # Strictly use the Mediapipe Tasks PoseLandmarker (no fallback).
        # NOTE: The Tasks API requires a model asset; if your environment does not
        # provide the default model, set `model_asset_path` in BaseOptions.
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="pose_landmarker_models\pose_landmarker_full.task"),
            running_mode=vision.RunningMode.VIDEO
        )
        pose_landmarker = PoseLandmarker.create_from_options(options)

        while self.running:
            image = self.frame_queue.get()
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(time.time() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            task_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            # Push the raw Tasks result directly; downstream code adjusted to read it.
            if not self.result_queue.empty():
                try: self.result_queue.get_nowait()
                except queue.Empty: pass
            self.result_queue.put((image, task_result, time.time()))

    def main_render_loop(self):
        while self.running:
            try:
                image, results, cur_time = self.result_queue.get(timeout=0.1)
            except queue.Empty: continue
            
            # The Mediapipe Tasks `PoseLandmarkerResult` may expose `pose_landmarks`
            # and `pose_world_landmarks` either as a LandmarkList or as a list
            # (batched). Normalize to the LandmarkList with `.landmark` for the
            # rest of the pipeline.
            s_attr = getattr(results, 'pose_landmarks', None)
            w_attr = getattr(results, 'pose_world_landmarks', None)
            if not s_attr or not w_attr:
                skip_processing = True
            else:
                skip_processing = False

            # Unwrap batched/list results if necessary
            if isinstance(s_attr, (list, tuple)) and len(s_attr) > 0:
                s_lm_list = s_attr[0].landmark
            else:
                s_lm_list = getattr(s_attr, 'landmark', None)

            if isinstance(w_attr, (list, tuple)) and len(w_attr) > 0:
                w_lm_list = w_attr[0].landmark
            else:
                w_lm_list = getattr(w_attr, 'landmark', None)

            if not s_lm_list or not w_lm_list:
                skip_processing = True

            s_lm = s_lm_list
            w_lm = w_lm_list

            s_lm = s_lm_list
            w_lm = w_lm_list
                
            # --- CALIBRATION ---
            if not self.is_calibrated:
                elapsed = cur_time - self.program_start_time
                if elapsed < self.COUNTDOWN_SECONDS:
                    cv2.putText(image, f"READY IN: {int(self.COUNTDOWN_SECONDS - elapsed)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    l_sh_w, r_sh_w = w_lm[11], w_lm[12]
                    l_sh_s, r_sh_s = s_lm[11], s_lm[12]

                    if l_sh_s.visibility > 0.5 and r_sh_s.visibility > 0.5:
                        self.fixed_sw_m = math.sqrt((l_sh_w.x-r_sh_w.x)**2 + (l_sh_w.y-r_sh_w.y)**2 + (l_sh_w.z-r_sh_w.z)**2)
                        fixed_sw_px = math.hypot((l_sh_s.x-r_sh_s.x)*self.frame_width, (l_sh_s.y-r_sh_s.y)*self.frame_height)
                        cam_dist_m = (self.fixed_sw_m * self.focal_length) / fixed_sw_px

                        self.kit.z_offset["L"] = w_lm[15].z / self.fixed_sw_m
                        self.kit.z_offset["R"] = w_lm[16].z / self.fixed_sw_m

                        anchor_x = int(((s_lm[23].x + s_lm[24].x) / 2) * self.frame_width)
                        anchor_y = int(((s_lm[23].y + s_lm[24].y) / 2) * self.frame_height)

                        self.cached_drum_positions = {}
                        for name, props in self.kit.drums.items():
                            drum_z_m = props["center"][2] * self.fixed_sw_m
                            depth_scale = cam_dist_m / (cam_dist_m + drum_z_m)

                            self.cached_drum_positions[name] = {
                                "cx": int(anchor_x + (props["center"][0] * fixed_sw_px * depth_scale)),
                                "cy": int(anchor_y + (props["center"][1] * fixed_sw_px * depth_scale)),
                                "rx": int((props["draw_radius"] * fixed_sw_px) * depth_scale),
                                "ry": int((props["draw_radius"] * fixed_sw_px * props["squash"]) * depth_scale)
                            }
                        self.is_calibrated = True

            # --- LIVE PROCESSING ---
            if self.is_calibrated:
                cur_time_ms = int(time.time() * 1000)
                dims = (self.frame_width, self.frame_height)

                hit_l, dbg_l = self.left_arm.process(s_lm[15], w_lm[15], s_lm[11], w_lm[11], s_lm[13], self.fixed_sw_m, self.kit, cur_time_ms, dims, s_lm[12])
                hit_r, dbg_r = self.right_arm.process(s_lm[16], w_lm[16], s_lm[12], w_lm[12], s_lm[14], self.fixed_sw_m, self.kit, cur_time_ms, dims, s_lm[11])

                # Update timestamps if a hit occurred
                if hit_l: self.last_l_hit_time = cur_time
                if hit_r: self.last_r_hit_time = cur_time

                # Display "Hand Hit!" messages if 'j' is active
                if self.show_hit_messages:
                    # Show message for 0.5 seconds after hit
                    if cur_time - self.last_l_hit_time < 0.5:
                        cv2.putText(image, "RIGHT HAND HIT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    if cur_time - self.last_r_hit_time < 0.5:
                        cv2.putText(image, "LEFT HAND HIT!", (self.frame_width - 350, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                # --- NEW: Constantly show Left Hand State ---
                if self.show_left_state:
                    # dbg_l["state"] contains either "UP" or "DOWN"
                    cv2.putText(image, f"LEFT STATE: {dbg_l['state']}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

                self._draw_arm_debug(image, dbg_l, (255, 0, 0))
                self._draw_arm_debug(image, dbg_r, (0, 0, 255))

                if self.show_drums and self.cached_drum_positions:
                    self._draw_drums(image, cur_time)

            cv2.imshow('AR Drum Gesture Mode', image)

            # Centralized key handling so the OpenCV window consistently receives input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.running = False
            elif key == ord('d'):
                self.show_drums = not self.show_drums
            elif key == ord('c'):
                self.show_coords = not self.show_coords
            elif key == ord('h'):
                self.show_occlusion = not self.show_occlusion
            elif key == ord('j'):
                self.show_hit_messages = not self.show_hit_messages
            elif key == ord('n'):
                self.show_drum_names = not self.show_drum_names
            elif key == ord('y'):
                self.show_left_state = not self.show_left_state  # <--- NEW

    def _draw_arm_debug(self, image, dbg, color):
        px = dbg["pos_px"]
        cv2.circle(image, px, 15, (0, 255, 0) if dbg["hit"] else color, -1)
        if self.show_coords:
            cv2.putText(image, f"STATE:{dbg['state']} Z:{dbg['z']:.2f}", (px[0]-40, px[1]-40), 0, 0.5, (255,255,255), 1)
        if self.show_occlusion and dbg["is_occluded"]:
            cv2.circle(image, dbg["sh_px"], int(0.15 * self.frame_width), (0, 0, 255), 2)

    def _draw_drums(self, image, cur_time):
        overlay = image.copy()
        
        # 1. Draw the filled, colored ellipses on the overlay
        for name, pos in self.cached_drum_positions.items():
            is_hit = (cur_time - self.kit.last_hit_time[name]) < 0.15
            color = (0, 255, 0) if is_hit else self.kit.drums[name]["color_idle"]
            cv2.ellipse(overlay, (pos["cx"], pos["cy"]), (pos["rx"], pos["ry"]), 0, 0, 360, color, -1)
            
        # 2. Blend the overlay to make the drums semi-transparent
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        # 3. Draw the Big Red Names directly on the final image (so they stay 100% solid)
        if self.show_drum_names:
            for name, pos in self.cached_drum_positions.items():
                # Calculate text size to perfectly center it on the drum
                text_size = cv2.getTextSize(name.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = pos["cx"] - (text_size[0] // 2)
                text_y = pos["cy"] + (text_size[1] // 2)
                
                # Draw a thick black outline first for readability
                cv2.putText(image, name.upper(), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                # Draw the bright red text over the outline
                cv2.putText(image, name.upper(), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Updated UI Help Text to include [N]
        cv2.putText(image, "[D] Drums [C] Coords [H] Occlusion [J] Hit Msg [N] Names", (10, self.frame_height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def start(self):
        threading.Thread(target=self.camera_thread, daemon=True).start()
        threading.Thread(target=self.ai_thread, daemon=True).start()
        self.main_render_loop()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ARDrumApp().start()