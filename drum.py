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
    
    def check_line_intersection(self, p1, p2, cur_time):
        """
        Check if the line segment from p1 to p2 intersects any drum cylinder.
        p1: (x, y, z) from the previous frame
        p2: (x, y, z) from the current frame
        """
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1

        for drum_name, props in self.drums.items():
            cx, cy, cz = props["center"]
            radius = props["hit_radius"]
            thickness = 0.15  # The vertical thickness of the drum
            drum_top = cz - (thickness / 2)
            drum_bottom = cz + (thickness / 2)

            # Translate to drum center
            x1_rel = x1 - cx
            y1_rel = y1 - cy

            # Quadratic coefficients for cylinder intersection: (x)^2 + (y)^2 = r^2
            a = dx**2 + dy**2
            b = 2 * (x1_rel * dx + y1_rel * dy)
            c = x1_rel**2 + y1_rel**2 - radius**2

            if a == 0:
                # Line is parallel to z-axis, check if starting point is inside cylinder
                if c <= 0 and drum_top <= z1 <= drum_bottom:
                    # Check cooldown
                    if cur_time - self.last_hit_time[drum_name] > self.hit_cooldown:
                        self.last_hit_time[drum_name] = cur_time
                        if drum_name in self.loaded_sounds:
                            self.loaded_sounds[drum_name].play()
                        return drum_name
                continue

            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                continue  # No intersection with infinite cylinder

            sqrt_d = math.sqrt(discriminant)
            t1 = (-b - sqrt_d) / (2 * a)
            t2 = (-b + sqrt_d) / (2 * a)

            # Check each intersection point
            for t in [t1, t2]:
                if 0 <= t <= 1:
                    # Compute z at intersection
                    z_intersect = z1 + t * dz
                    if drum_top <= z_intersect <= drum_bottom:
                        # Check cooldown
                        if cur_time - self.last_hit_time[drum_name] > self.hit_cooldown:
                            self.last_hit_time[drum_name] = cur_time
                            if drum_name in self.loaded_sounds:
                                self.loaded_sounds[drum_name].play()
                            return drum_name

        return None

    def cleanup(self):
        """Call this when your app closes to safely shut down the audio engine."""
        pygame.quit()