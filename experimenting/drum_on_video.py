import cv2
import mediapipe as mp
import time
import math

class VirtualDrumKit:
    def __init__(self):
        # Memory for smoothing and the resting 'zero' offset
        self.z_memory = {"L": 0.0, "R": 0.0}
        self.z_offset = {"L": 0.0, "R": 0.0} 
        
        # Expanded drum dictionary including visuals
        self.drums = {
            "Snare": {"center": (0.0, -0.30, -0.60), "radius": 0.41, "squash": 0.35, "color_idle": (200, 200, 200)},
            "Bass Drum": {"center": (0.35, 0.40, -0.80), "radius": 0.65, "squash": 0.85, "color_idle": (50, 50, 50)},
            "Hi-Hat": {"center": (-0.50, -0.25, -0.65), "radius": 0.41, "squash": 0.30, "color_idle": (0, 200, 255)},
            "High Tom": {"center": (-0.15, -0.70, -0.75), "radius": 0.35, "squash": 0.60, "color_idle": (255, 100, 100)},
            "Mid Tom": {"center": (0.15, -0.70, -0.75), "radius": 0.38, "squash": 0.60, "color_idle": (255, 100, 100)},
            "Floor Tom": {"center": (0.75, -0.20, -0.50), "radius": 0.47, "squash": 0.35, "color_idle": (200, 100, 100)},
            "Ride Cymbal": {"center": (0.60, -0.80, -1.60), "radius": 0.59, "squash": 0.45, "color_idle": (0, 215, 255)},
            "Crash Cymbal": {"center": (-0.75, -1.20, -1), "radius": 0.47, "squash": 0.45, "color_idle": (0, 215, 255)}
        }

    def check_hit_old(self, x, y, z):
        for drum_name, props in self.drums.items():
            cx, cy, cz = props["center"]
            radius = props["radius"]
            distance = math.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
            if distance <= radius:
                return drum_name
        return None
    
    def check_hit(self, x, y, z):
        for drum_name, props in self.drums.items():
            cx, cy, cz = props["center"]
            radius = props["radius"]
            
            # 1. Check 2D Surface Distance (X and Y only)
            # Are you hovering over the drum head?
            surface_distance = math.sqrt((x - cx)**2 + (y - cy)**2)
            
            # 2. Check 1D Depth Distance (Z only)
            # How deep is your stick currently penetrating?
            depth_distance = abs(z - cz)
            
            # Define how thick the drum is (e.g., 0.15 is about 6 inches thick)
            # You can tweak this number to make the cymbals thinner or thicker!
            drum_thickness = 0.15 
            
            # IT'S A HIT ONLY IF:
            # You are within the wide X/Y radius AND inside the thin Z thickness
            if surface_distance <= radius and depth_distance <= drum_thickness:
                return drum_name
                
        return None

# --- Math Helper for Angles ---
def calculate_3d_angle(a, b, c):
    """Calculates the angle ABC (with B as the vertex) in 3D space."""
    # Vector BA (Shoulder to Elbow)
    ba = [a.x - b.x, a.y - b.y, a.z - b.z]
    # Vector BC (Wrist to Elbow)
    bc = [c.x - b.x, c.y - b.y, c.z - b.z]
    
    # Dot product and magnitudes
    dot_product = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2 + ba[2]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2 + bc[2]**2)
    
    if mag_ba * mag_bc == 0: return 0.0
    
    # Clamp value to avoid math domain errors due to floating point inaccuracies
    cos_angle = max(-1.0, min(1.0, dot_product / (mag_ba * mag_bc)))
    angle_rad = math.acos(cos_angle)
    
    return math.degrees(angle_rad)

def calculate_2d_angle(a, b, c):
    """Calculates the 2D angle ABC on the screen, ignoring Z-depth."""
    # Calculate the angle using atan2
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle = math.degrees(radians)
    
    # Ensure the angle is always a positive number between 0 and 180
    angle = abs(angle)
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

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
show_drums = True # Toggle state for drums
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
                            
                            l_wrist_wrl = world_lm[mp_holistic.PoseLandmark.LEFT_WRIST]
                            r_wrist_wrl = world_lm[mp_holistic.PoseLandmark.RIGHT_WRIST]
                            my_kit.z_offset["L"] = l_wrist_wrl.z / fixed_sw_m
                            my_kit.z_offset["R"] = r_wrist_wrl.z / fixed_sw_m
                            
                            is_calibrated = True
                    else:
                        program_start_time = time.time() - COUNTDOWN_SECONDS + 0.1

            # --- 2. WRIST TRACKING & DYNAMIC SMOOTHING ---
            active_drums = set()

            def process_wrist(landmark_index, shoulder_index, elbow_index, color, label):
                w_scr = screen_lm[landmark_index]
                w_wrl = world_lm[landmark_index]
                
                sh_scr = screen_lm[shoulder_index]
                sh_wrl = world_lm[shoulder_index]
                
                el_scr = screen_lm[elbow_index] # We need the elbow now!
                
                if w_scr.visibility > 0.5:
                    cx, cy = int(w_scr.x * frame_width), int(w_scr.y * frame_height)
                    
                    # --- UPGRADED: Smart Occlusion Dampener ---
                    # Distance between wrist and shoulder
                    dist_wrist_shoulder = math.hypot(w_scr.x - sh_scr.x, w_scr.y - sh_scr.y)
                    # Distance between elbow and shoulder
                    dist_elbow_shoulder = math.hypot(el_scr.x - sh_scr.x, el_scr.y - sh_scr.y)
                    
                    corrected_z_wrl = w_wrl.z 
                    
                    # IF wrist is near shoulder AND elbow is far (arm is folded) -> DAMPEN
                    # IF elbow is also near (punching at camera) -> DO NOT DAMPEN
                    if dist_wrist_shoulder < 0.15 and dist_elbow_shoulder > 0.15: 
                        dampening = dist_wrist_shoulder / 0.15
                        corrected_z_wrl = sh_wrl.z + ((w_wrl.z - sh_wrl.z) * dampening)
                    # -------------------------------
                    
                    # 1. Normalize and Apply Tare Offset 
                    raw_x = w_wrl.x / fixed_sw_m
                    raw_y = w_wrl.y / fixed_sw_m
                    raw_z_tared = (corrected_z_wrl / fixed_sw_m) - my_kit.z_offset[label]
                    
                    # 2. Dynamic Smoothing Logic
                    delta_z = abs(raw_z_tared - my_kit.z_memory[label])
                    
                    if delta_z < 0.02: current_smoothing = 0.85
                    elif delta_z > 0.15: current_smoothing = 0.10
                    else:
                        progress = (delta_z - 0.02) / (0.15 - 0.02)
                        current_smoothing = 0.85 - (progress * 0.75)
                        
                    smooth_z = (my_kit.z_memory[label] * current_smoothing) + (raw_z_tared * (1.0 - current_smoothing))
                    my_kit.z_memory[label] = smooth_z

                    # 3. Hit Detection
                    hit = my_kit.check_hit(raw_x, raw_y, smooth_z)
                    
                    # 4. Draw Wrist Pointer
                    is_striking = current_smoothing < 0.2
                    if hit: p_color = (0, 255, 0)
                    elif is_striking: p_color = (255, 255, 255)
                    else: p_color = color
                        
                    cv2.circle(image, (cx, cy), 15, p_color, -1)
                    
                    # 5. Overlay Text
                    depth_text = f"Z: {smooth_z:.2f} (Smth: {current_smoothing:.2f})"
                    cv2.putText(image, depth_text, (cx - 40, cy - 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    return hit
                return None
            
            if is_calibrated:
                hit_l = process_wrist(
                    mp_holistic.PoseLandmark.LEFT_WRIST, 
                    mp_holistic.PoseLandmark.LEFT_SHOULDER, 
                    mp_holistic.PoseLandmark.LEFT_ELBOW, # Added Elbow
                    (255, 0, 0), "L"
                )
                
                hit_r = process_wrist(
                    mp_holistic.PoseLandmark.RIGHT_WRIST, 
                    mp_holistic.PoseLandmark.RIGHT_SHOULDER, 
                    mp_holistic.PoseLandmark.RIGHT_ELBOW, # Added Elbow
                    (0, 0, 255), "R"
                )
                if hit_l: active_drums.add(hit_l)
                if hit_r: active_drums.add(hit_r)

            # --- 3. LIVE DISTANCE, ANCHOR & ANGLES ---
            if is_calibrated:
                # Update Distance & Anchors
                l_sh_scr = screen_lm[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                r_sh_scr = screen_lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                
                live_sw_px = math.hypot((l_sh_scr.x - r_sh_scr.x) * frame_width, 
                                        (l_sh_scr.y - r_sh_scr.y) * frame_height)
                
                if live_sw_px > 0:
                    camera_distance_m = (fixed_sw_m * approx_focal_length) / live_sw_px

                l_hip_scr = screen_lm[mp_holistic.PoseLandmark.LEFT_HIP]
                r_hip_scr = screen_lm[mp_holistic.PoseLandmark.RIGHT_HIP]
                anchor_x_px = int(((l_hip_scr.x + r_hip_scr.x) / 2) * frame_width)
                anchor_y_px = int(((l_hip_scr.y + r_hip_scr.y) / 2) * frame_height)

                # --- FIXED: 2D Arm Angle Calculations ---
                l_sh_scr_ang = screen_lm[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                l_el_scr_ang = screen_lm[mp_holistic.PoseLandmark.LEFT_ELBOW]
                l_wr_scr_ang = screen_lm[mp_holistic.PoseLandmark.LEFT_WRIST]
                
                r_sh_scr_ang = screen_lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                r_el_scr_ang = screen_lm[mp_holistic.PoseLandmark.RIGHT_ELBOW]
                r_wr_scr_ang = screen_lm[mp_holistic.PoseLandmark.RIGHT_WRIST]

                # Calculate 2D Screen Angles
                angle_l = calculate_2d_angle(l_sh_scr_ang, l_el_scr_ang, l_wr_scr_ang)
                angle_r = calculate_2d_angle(r_sh_scr_ang, r_el_scr_ang, r_wr_scr_ang)

                # Draw Left Angle near left elbow
                l_el_scr = screen_lm[mp_holistic.PoseLandmark.LEFT_ELBOW]
                if l_el_scr.visibility > 0.5:
                    ex, ey = int(l_el_scr.x * frame_width), int(l_el_scr.y * frame_height)
                    cv2.putText(image, f"{int(angle_l)} deg", (ex + 15, ey), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Draw Right Angle near right elbow
                r_el_scr = screen_lm[mp_holistic.PoseLandmark.RIGHT_ELBOW]
                if r_el_scr.visibility > 0.5:
                    ex, ey = int(r_el_scr.x * frame_width), int(r_el_scr.y * frame_height)
                    cv2.putText(image, f"{int(angle_r)} deg", (ex - 80, ey), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # --- 4. DRAW PERSPECTIVE DRUM KIT ---
            if is_calibrated and show_drums:
                overlay = image.copy()
                
                for drum_name, props in my_kit.drums.items():
                    drum_z_m = props["center"][2] * fixed_sw_m
                    
                    if camera_distance_m + drum_z_m <= 0.1: depth_scale = 10.0 
                    else: depth_scale = camera_distance_m / (camera_distance_m + drum_z_m)
                    
                    center_x_px = int(anchor_x_px + (props["center"][0] * fixed_sw_px * depth_scale))
                    center_y_px = int(anchor_y_px + (props["center"][1] * fixed_sw_px * depth_scale))
                    
                    base_radius_px = props["radius"] * fixed_sw_px
                    radius_px = int(base_radius_px * depth_scale)
                    squashed_radius_px = int(radius_px * props["squash"])
                    
                    if drum_name in active_drums:
                        color = (0, 255, 0); thickness = -1      
                    else:
                        color = props["color_idle"]; thickness = 3       

                    cv2.ellipse(overlay, (center_x_px, center_y_px), (radius_px, squashed_radius_px), 0, 0, 360, color, thickness)
                    if thickness > 0: cv2.circle(overlay, (center_x_px, center_y_px), 2, color, -1)

                    cv2.putText(overlay, drum_name, (center_x_px - 20, center_y_px + squashed_radius_px + 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

            # --- 5. UI OVERLAY ---
            if active_drums:
                hit_text = " | ".join(list(active_drums))
                cv2.putText(image, f"HIT: {hit_text}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            if is_calibrated:
                cv2.putText(image, f"Distance: {camera_distance_m:.2f}m", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image, "Press 'd' to toggle drums", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        cv2.imshow('AR Drum Logic', image)
        
        # Keyboard listener
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC key
            break
        elif key == ord('d'): # Toggle drums
            show_drums = not show_drums

cap.release()
cv2.destroyAllWindows()