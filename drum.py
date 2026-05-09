import math
import os
import pygame # --- NEW: Import PyGame ---

class VirtualDrumKit:
    def __init__(self):
        self.z_memory = {"L": 0.0, "R": 0.0}
        self.z_offset = {"L": 0.0, "R": 0.0} 
        
        self.last_hit_time = {}
        self.hit_cooldown = 0.2
        
        # --- NEW: Initialize Audio Mixer ---
        # buffer=512 ensures ultra-low latency so the sound triggers exactly when you punch
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.init()
        pygame.mixer.set_num_channels(16) # Allows up to 16 sounds to overlap at once
        
        # --- UPGRADED: Added 'sound_path' to each drum ---
        self.drums = {
            "Snare": {"center": (0.0, -0.30, -0.45), "draw_radius": 0.41, "hit_radius": 0.35, "squash": 0.35, "pitch": 10, "color_idle": (200, 200, 200), "sound_path": "sounds/Snare Sample.mp3"},
           # "Bass Drum": {"center": (0.35, 0.40, -0.55), "draw_radius": 0.65, "hit_radius": 0.35, "squash": 0.85, "pitch": 0, "color_idle": (50, 50, 50), "sound_path": "sounds/kick.wav"},
            "Hi-Hat": {"center": (-0.90, -0.25, -0.55), "draw_radius": 0.41, "hit_radius": 0.35, "squash": 0.30, "pitch": 5, "color_idle": (0, 200, 255), "sound_path": "sounds/HI-HAT Top Sample.mp3"},
            "High Tom": {"center": (-0.45, -0.73, -0.65), "draw_radius": 0.25, "hit_radius": 0.28, "squash": 0.60, "pitch": 25, "color_idle": (255, 100, 100), "sound_path": "sounds/High Tom Sample.mp3"},
            "Mid Tom": {"center": (0.32, -0.70, -0.65), "draw_radius": 0.33, "hit_radius": 0.28, "squash": 0.60, "pitch": 25, "color_idle": (255, 100, 100), "sound_path": "sounds/Middle Tom Sample.mp3"},
           # "Floor Tom": {"center": (1.10, -0.20, -0.50), "draw_radius": 0.47, "hit_radius": 0.25, "squash": 0.35, "pitch": 10, "color_idle": (200, 100, 100), "sound_path": "sounds/tom_floor.wav"},
            "Ride Cymbal": {"center": (1.10, -0.80, -0.60), "draw_radius": 0.54, "hit_radius": 0.30, "squash": 0.45, "pitch": 15, "color_idle": (0, 215, 255), "sound_path": "sounds/Ride Cymbal Edge Sample.mp3"},
            "Crash Cymbal": {"center": (-1.10, -1.05, -0.85), "draw_radius": 0.47, "hit_radius": 0.28, "squash": 0.45, "pitch": 15, "color_idle": (0, 215, 255), "sound_path": "sounds/High Crash Cymbal Sample.mp3"}
        }
        
        # --- NEW: Load sounds into memory ---
        self.loaded_sounds = {}
        for drum_name, props in self.drums.items():
            self.last_hit_time[drum_name] = 0.0
            
            # Check if file exists, then load it
            path = props["sound_path"]
            if os.path.exists(path):
                self.loaded_sounds[drum_name] = pygame.mixer.Sound(path)
            else:
                print(f"WARNING: Missing audio file -> {path}")

    def check_hit(self, x, y, z, current_time):
        for drum_name, props in self.drums.items():
            cx, cy, cz = props["center"]
            radius = props["hit_radius"] 
            
            # 1. Translate the hand point to the drum's local space
            dx = x - cx
            dy = y - cy
            dz = z - cz
            
            # 2. Get the drum's physical pitch angle
            pitch_rad = math.radians(props["pitch"])
            
            # 3. ARCADE PITCH ROTATION (Rotate around X-axis)
            local_x = dx
            local_y = dy * math.cos(pitch_rad) - dz * math.sin(pitch_rad)
            local_z = dy * math.sin(pitch_rad) + dz * math.cos(pitch_rad)
            
            # 4. Check Flat Collision in Local Tilted Space
            surface_distance = math.sqrt(local_x**2 + local_y**2)
            depth_distance = abs(local_z)
            drum_thickness = 0.25 
            
            # IT'S A HIT ONLY IF:
            if surface_distance <= radius and depth_distance <= drum_thickness:
                if current_time - self.last_hit_time[drum_name] > self.hit_cooldown:
                    self.last_hit_time[drum_name] = current_time
                    
                    # --- NEW: Play the sound instantly! ---
                    if drum_name in self.loaded_sounds:
                        self.loaded_sounds[drum_name].play()
                        
                    return drum_name
        return None
    
    # Add this inside your VirtualDrumKit class in drum.py
    def check_line_intersection(self, p1, p2, cur_time):
        """
        p1: (x, y, z) from the previous frame
        p2: (x, y, z) from the current frame
        """
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        for drum_name, props in self.drums.items():
            cx, cy, cz = props["center"]
            radius = props["hit_radius"]
            thickness = 0.15 # The vertical thickness of the drum

            # 1. Did the line cross the Z-depth of this drum?
            min_z = min(z1, z2)
            max_z = max(z1, z2)
            drum_top = cz - (thickness / 2)
            drum_bottom = cz + (thickness / 2)

            if max_z < drum_top or min_z > drum_bottom:
                continue # The line is completely above or below the drum.

            # 2. Calculate exactly where the line crossed the drum's Z-plane.
            # We find the parameter 't' (from 0.0 to 1.0) along the line segment.
            if z2 != z1:
                t = (cz - z1) / (z2 - z1)
                t = max(0.0, min(1.0, t)) # Clamp to the segment length
            else:
                t = 1.0

            # 3. Find the exact X, Y coordinates at that point of crossing
            intersect_x = x1 + t * (x2 - x1)
            intersect_y = y1 + t * (y2 - y1)

            # 4. Check if those X, Y coordinates are inside the drum's radius
            surface_dist = math.hypot(intersect_x - cx, intersect_y - cy)

            if surface_dist <= radius:
                # Add the cooldown check just to be safe
                if cur_time - self.last_hit_time[drum_name] > self.hit_cooldown:
                    self.last_hit_time[drum_name] = cur_time
                    
                    # --- THE MISSING AUDIO TRIGGER ---
                    if drum_name in self.loaded_sounds:
                        self.loaded_sounds[drum_name].play()
                    # ---------------------------------
                return drum_name
                
        return None

    def cleanup(self):
        """Call this when your app closes to safely shut down the audio engine."""
        pygame.quit()