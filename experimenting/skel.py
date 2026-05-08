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

# --- DISPLAY FLAG ---
# Set to 1: Shows Face Mesh + Red/Blue dots on wrists + XYZ Coordinates.
# Set to 0: Shows Face Mesh + Full Body Skeleton + Hand Skeleton.
DOTS_ONLY = 0
# Initialize video capture
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

# Get video properties for the VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

print(f"Processing video: {frame_width}x{frame_height} at {fps} FPS...")

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1 
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
        
        # --- 1. DRAW FULL FACE MESH (Always visible) ---
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

        # --- 2. TOGGLE BETWEEN DOTS OR FULL SKELETON ---
        if DOTS_ONLY == 1:
            # Draw ONLY specific dots and their coordinates
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # 1. Get Shoulder Landmarks
                l_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                r_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                
                # Convert normalized (0.0 - 1.0) to absolute pixels
                l_sh_x, l_sh_y = (l_shoulder.x * frame_width, l_shoulder.y * frame_height)
                r_sh_x, r_sh_y = (r_shoulder.x * frame_width, r_shoulder.y * frame_height)
                
                # 2. Calculate Shoulder Width in pixels using math.hypot (Pythagorean theorem)
                shoulder_width_px = math.hypot(l_sh_x - r_sh_x, l_sh_y - r_sh_y)
                
                # Safety check: prevent division by zero if shoulders perfectly overlap (rare)
                if shoulder_width_px == 0:
                    shoulder_width_px = 1 

                # Helper function to draw a point and its coordinates
                def draw_point(landmark_index, color, label):
                    point = landmarks[landmark_index]
                    if point.visibility > 0.5:
                        cx = int(point.x * frame_width)
                        cy = int(point.y * frame_height)
                        
                        # Draw the dot
                        cv2.circle(image, (cx, cy), 8, color, -1)


                        # 3. Get the Wrist and calculate Body-Relative Depth
                        wrist = landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST]
                        
                        # MediaPipe's Z is scaled to frame width. Convert to "Z-Pixels"
                        wrist_z_px = wrist.z * frame_width
                        
                        # Divide by our dynamic ruler
                        relative_z = wrist_z_px / shoulder_width_px

                        wrist_x_px = wrist.x * frame_width
                        
                        # Divide by our dynamic ruler
                        relative_x = wrist_x_px / shoulder_width_px
                        wrist_y_px = wrist.y * frame_width
                        
                        # Divide by our dynamic ruler
                        relative_y = wrist_y_px / shoulder_width_px
                        # Format the text (X, Y, Z to 2 decimal places)
                        coord_text = f"{label} (X:{relative_x:.2f} Y:{relative_y:.2f} Z:{relative_z:.2f})"
                        
                        # Draw the text slightly offset (15px right, 10px up) from the dot
                        cv2.putText(image, coord_text, (cx - 25, cy - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw Left Wrist (Red) and Right Wrist (Blue)
                draw_point(mp_holistic.PoseLandmark.LEFT_WRIST, (0, 0, 255), "L")
                draw_point(mp_holistic.PoseLandmark.RIGHT_WRIST, (255, 0, 0), "R")
                
        else:
            # Draw FULL MediaPipe Skeleton
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            # Add full finger tracking if visible
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    results.left_hand_landmarks, 
                    mp_holistic.HAND_CONNECTIONS
                )
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    results.right_hand_landmarks, 
                    mp_holistic.HAND_CONNECTIONS
                )
           
  
            landmarks = results.pose_landmarks.landmark
            
            # 1. Get Shoulder Landmarks
            l_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            r_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            
            # Convert normalized (0.0 - 1.0) to absolute pixels
            l_sh_x, l_sh_y = (l_shoulder.x * frame_width, l_shoulder.y * frame_height)
            r_sh_x, r_sh_y = (r_shoulder.x * frame_width, r_shoulder.y * frame_height)
            
            # 2. Calculate Shoulder Width in pixels using math.hypot (Pythagorean theorem)
            shoulder_width_px = math.hypot(l_sh_x - r_sh_x, l_sh_y - r_sh_y)
            
            # Safety check: prevent division by zero if shoulders perfectly overlap (rare)
            if shoulder_width_px == 0:
                shoulder_width_px = 1 

            # 3. Get the Wrist and calculate Body-Relative Depth
            wrist = landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST]
            
            # MediaPipe's Z is scaled to frame width. Convert to "Z-Pixels"
            wrist_z_px = wrist.z * frame_width
            
            # Divide by our dynamic ruler
            relative_z = wrist_z_px / shoulder_width_px
            
            print(f"Right Wrist Relative Z: {relative_z:.2f} shoulder-widths")
        # -------------------------------------------

        # Calculate and print metrics
        latency = (end_time - start_time) * 1000 
        processing_fps = 1 / (end_time - start_time)
        print(f"Processing Frame Latency: {latency:.2f}ms | Processing FPS: {processing_fps:.2f}")

        # Write the annotated frame to the output file
        out.write(image)

        # Display the output while processing
        cv2.imshow('MediaPipe Tracking', image)
        if cv2.waitKey(1) & 0xFF == 27: # Press ESC to cancel early
            print("Processing cancelled by user.")
            break

# Clean up resources
cap.release()
out.release()
cv2.destroyAllWindows()