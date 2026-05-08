import cv2
import mediapipe as mp
import time
import math

class VirtualDrumKit:
    def __init__(self):
        # Memory for smoothing and the resting 'zero' offset
        self.z_memory = {"L": 0.0, "R": 0.0}
        self.z_offset = {"L": 0.0, "R": 0.0} 
        self.drums = {
            "Snare": {"center": (0.0, -0.30, -0.60), "radius": 0.41},
            "Bass Drum": {"center": (0.35, 0.40, -0.80), "radius": 0.65},
            "Hi-Hat": {"center": (-0.50, -0.25, -0.65), "radius": 0.41},
            "High Tom": {"center": (-0.15, -0.70, -0.75), "radius": 0.35},
            "Mid Tom": {"center": (0.15, -0.70, -0.75), "radius": 0.38},
            "Floor Tom": {"center": (0.75, -0.20, -0.50), "radius": 0.47},
            "Ride Cymbal": {"center": (0.60, -0.80, -0.60), "radius": 0.59},
            "Crash Cymbal": {"center": (-0.75, -1.20, -1), "radius": 0.47}
        }

    def check_hit(self, x, y, z):
        for drum_name, props in self.drums.items():
            cx, cy, cz = props["center"]
            radius = props["radius"]
            distance = math.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
            if distance <= radius:
                return drum_name
        return None

# --- Configuration ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ASSUMED_FOV_DEG = 65.0
fov_radians = math.radians(ASSUMED_FOV_DEG)
approx_focal_length = (frame_width / 2) / math.tan(fov_radians / 2)

my_kit = VirtualDrumKit()

is_calibrated = False
fixed_sw_m = 1.0   
fixed_sw_px = 100
camera_distance_m = 1.0 

program_start_time = time.time()
COUNTDOWN_SECONDS = 5

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1 
) as holistic:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success: break
        image = cv2.flip(image, 1)

        image.flags.writeable = False
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image.flags.writeable = True
        
        if results.pose_landmarks and results.pose_world_landmarks:
            screen_lm = results.pose_landmarks.landmark
            world_lm = results.pose_world_landmarks.landmark
            
            # --- 1. CALIBRATION ---
            if not is_calibrated:
                elapsed_time = time.time() - program_start_time
                if elapsed_time < COUNTDOWN_SECONDS:
                    time_left = int(math.ceil(COUNTDOWN_SECONDS - elapsed_time))
                    cv2.putText(image, f"CALIBRATING IN: {time_left}", (frame_width//4, frame_height//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    l_sh_scr = screen_lm[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                    r_sh_scr = screen_lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                    l_sh_wrl = world_lm[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                    r_sh_wrl = world_lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                    
                    if l_sh_scr.visibility > 0.5 and r_sh_scr.visibility > 0.5:
                        fixed_sw_m = math.sqrt((l_sh_wrl.x-r_sh_wrl.x)**2 + (l_sh_wrl.y-r_sh_wrl.y)**2 + (l_sh_wrl.z-r_sh_wrl.z)**2)
                        fixed_sw_px = math.hypot((l_sh_scr.x-r_sh_scr.x)*frame_width, (l_sh_scr.y-r_sh_scr.y)*frame_height)
                        if fixed_sw_px > 0:
                            camera_distance_m = (fixed_sw_m * approx_focal_length) / fixed_sw_px
                            
                            # --- NEW: Capture Resting Wrist Positions (Tare) ---
                            l_wrist_wrl = world_lm[mp_holistic.PoseLandmark.LEFT_WRIST]
                            r_wrist_wrl = world_lm[mp_holistic.PoseLandmark.RIGHT_WRIST]
                            my_kit.z_offset["L"] = l_wrist_wrl.z / fixed_sw_m
                            my_kit.z_offset["R"] = r_wrist_wrl.z / fixed_sw_m
                            # ---------------------------------------------------
                            
                            is_calibrated = True
                    else:
                        program_start_time = time.time() - COUNTDOWN_SECONDS + 0.1

            # --- 2. WRIST TRACKING ---
            active_hit = None

            def process_wrist(landmark_index, color, label):
                w_scr = screen_lm[landmark_index]
                w_wrl = world_lm[landmark_index]
                
                if w_scr.visibility > 0.5:
                    cx, cy = int(w_scr.x * frame_width), int(w_scr.y * frame_height)
                    
                    # 1. Normalize and Apply Tare Offset
                    raw_x = w_wrl.x / fixed_sw_m
                    raw_y = w_wrl.y / fixed_sw_m
                    raw_z_tared = (w_wrl.z / fixed_sw_m) - my_kit.z_offset[label]
                    
                    # 2. Dynamic Smoothing Logic
                    delta_z = abs(raw_z_tared - my_kit.z_memory[label])
                    
                    if delta_z < 0.02:
                        current_smoothing = 0.85
                    elif delta_z > 0.15:
                        current_smoothing = 0.10
                    else:
                        progress = (delta_z - 0.02) / (0.15 - 0.02)
                        current_smoothing = 0.85 - (progress * 0.75)
                        
                    smooth_z = (my_kit.z_memory[label] * current_smoothing) + (raw_z_tared * (1.0 - current_smoothing))
                    my_kit.z_memory[label] = smooth_z

                    # 3. Hit Detection
                    hit = my_kit.check_hit(raw_x, raw_y, smooth_z)
                    
                    # 4. Draw Wrist Pointer (White for strike, Green for hit, Base color for idle)
                    is_striking = current_smoothing < 0.2
                    if hit:
                        p_color = (0, 255, 0)
                    elif is_striking:
                        p_color = (255, 255, 255)
                    else:
                        p_color = color
                        
                    cv2.circle(image, (cx, cy), 15, p_color, -1)
                    
                    # 5. Overlay Text
                    vis_val = w_scr.visibility
                    pres_val = getattr(w_scr, 'presence', 0.0)
                    
                    depth_text = f"Z: {smooth_z:.2f} (Smth: {current_smoothing:.2f})"
                    vis_pres_text = f"Vis: {vis_val:.2f} | Pres: {pres_val:.2f}"
                    
                    cv2.putText(image, depth_text, (cx - 40, cy - 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(image, vis_pres_text, (cx - 40, cy - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    
                    return hit
                return None

            if is_calibrated:
                hit_l = process_wrist(mp_holistic.PoseLandmark.LEFT_WRIST, (255, 0, 0), "L")
                hit_r = process_wrist(mp_holistic.PoseLandmark.RIGHT_WRIST, (0, 0, 255), "R")
                active_hit = hit_l or hit_r

            # --- 3. UI OVERLAY ---
            if active_hit:
                cv2.putText(image, f"HIT: {active_hit}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            if is_calibrated:
                cv2.putText(image, f"Distance: {camera_distance_m:.2f}m", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        cv2.imshow('AR Drum Logic', image)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()