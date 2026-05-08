import cv2
import mediapipe as mp
import time
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# --- Configuration ---
INPUT_VIDEO_PATH = 'air_drumming.mp4'   # Replace with your video file name
OUTPUT_VIDEO_PATH = 'output_video.mp4'  # Name of the saved file
DOTS_ONLY = 1

# Initialize video capture
cap = cv2.VideoCapture(0)

# Get video properties for the VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

print(f"Processing video: {frame_width}x{frame_height} at {fps} FPS...")

# --- NEW: CALIBRATION VARIABLES ---
is_calibrated = False
fixed_shoulder_width_px = 1.0 # Default safe value to prevent math errors

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=2
) as holistic:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Video processing complete.")
            break 

        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        start_time = time.perf_counter()
        results = holistic.process(image_rgb)
        end_time = time.perf_counter()

        image.flags.writeable = True
        
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

        if DOTS_ONLY == 1:
            # We MUST check that BOTH the 2D drawing landmarks and 3D math landmarks exist
            if results.pose_landmarks and results.pose_world_landmarks:
                
                # We use these purely for drawing circles on the screen
                screen_landmarks = results.pose_landmarks.landmark
                
                # We use these purely for perfectly scaled 3D math (in meters)
                world_landmarks = results.pose_world_landmarks.landmark
                
                # --- 1. FIRST FRAME CALIBRATION (THE RULER IN METERS) ---
                if not is_calibrated:
                    l_sh_world = world_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                    r_sh_world = world_landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                    
                    if l_sh_world.visibility > 0.5 and r_sh_world.visibility > 0.5:
                        # Calculate the 3D distance between shoulders in real-world METERS
                        fixed_shoulder_width_m = math.sqrt(
                            (l_sh_world.x - r_sh_world.x)**2 + 
                            (l_sh_world.y - r_sh_world.y)**2 + 
                            (l_sh_world.z - r_sh_world.z)**2
                        )
                        
                        # A standard adult shoulder width is between 0.35m and 0.5m
                        if fixed_shoulder_width_m > 0.1: 
                            is_calibrated = True
                            print(f"Calibration Locked! 1 Unit = {fixed_shoulder_width_m:.2f} meters")
                
                if is_calibrated:
                    cv2.putText(image, f"CALIBRATED: 1 SW = {fixed_shoulder_width_m:.2f}m", 
                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Helper function to draw a point and its coordinates
                def draw_point(landmark_index, color, label):
                    # Get the 2D point to know where to draw
                    screen_pt = screen_landmarks[landmark_index]
                    
                    # Get the 3D point to know the true physical location
                    world_pt = world_landmarks[landmark_index]
                    
                    if screen_pt.visibility > 0.5:
                        # Draw the dot on the 2D video feed
                        cx = int(screen_pt.x * frame_width)
                        cy = int(screen_pt.y * frame_height)
                        cv2.circle(image, (cx, cy), 8, color, -1)

                        # --- THE PERFECT MATH ---
                        # Because world_pt uses the hips as (0,0,0) by default, 
                        # all we have to do is divide by our physical meter ruler.
                        relative_x = world_pt.x / fixed_shoulder_width_m
                        relative_y = world_pt.y / fixed_shoulder_width_m
                        relative_z = world_pt.z / fixed_shoulder_width_m

                        coord_text = f"{label} (X:{relative_x:.2f} Y:{relative_y:.2f} Z:{relative_z:.2f})"
                        cv2.putText(image, coord_text, (cx - 75, cy - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw Left Wrist (Red) and Right Wrist (Blue)
                draw_point(mp_holistic.PoseLandmark.LEFT_WRIST, (0, 0, 255), "L")
                draw_point(mp_holistic.PoseLandmark.RIGHT_WRIST, (255, 0, 0), "R")
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
        else:
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

        latency = (end_time - start_time) * 1000 
        processing_fps = 1 / (end_time - start_time)

        out.write(image)
        cv2.imshow('Air Drum Calibration', image)
        if cv2.waitKey(1) & 0xFF == 27: 
            break

cap.release()
out.release()
cv2.destroyAllWindows()