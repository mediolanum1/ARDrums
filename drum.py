import math
import os
import pygame # --- NEW: Import PyGame ---

class VirtualDrumKit:
    def __init__(self):
        self.z_memory = {"L": 0.0, "R": 0.0}
        self.z_offset = {"L": 0.0, "R": 0.0} 
        
        self.last_hit_time = {}
        self.hit_cooldown = 0.25 
        
        # --- NEW: Initialize Audio Mixer ---
        # buffer=512 ensures ultra-low latency so the sound triggers exactly when you punch
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.init()
        pygame.mixer.set_num_channels(16) # Allows up to 16 sounds to overlap at once
        
        # --- UPGRADED: Added 'sound_path' to each drum ---
        self.drums = {
            "Snare": {"center": (0.0, -0.30, -0.40), "draw_radius": 0.41, "hit_radius": 0.25, "squash": 0.35, "pitch": 10, "color_idle": (200, 200, 200), "sound_path": "sounds/Snare Sample.mp3"},
           # "Bass Drum": {"center": (0.35, 0.40, -0.55), "draw_radius": 0.65, "hit_radius": 0.35, "squash": 0.85, "pitch": 0, "color_idle": (50, 50, 50), "sound_path": "sounds/kick.wav"},
            "Hi-Hat": {"center": (-0.80, -0.25, -0.75), "draw_radius": 0.41, "hit_radius": 0.25, "squash": 0.30, "pitch": 5, "color_idle": (0, 200, 255), "sound_path": "sounds/HI-HAT.mp3"},
            "High Tom": {"center": (-0.25, -0.70, -0.85), "draw_radius": 0.35, "hit_radius": 0.20, "squash": 0.60, "pitch": 25, "color_idle": (255, 100, 100), "sound_path": "sounds/High Tim Sample.mp3"},
            "Mid Tom": {"center": (0.25, -0.70, -0.85), "draw_radius": 0.38, "hit_radius": 0.20, "squash": 0.60, "pitch": 25, "color_idle": (255, 100, 100), "sound_path": "sounds/Middle Tom Sample.mp3"},
           # "Floor Tom": {"center": (1.10, -0.20, -0.50), "draw_radius": 0.47, "hit_radius": 0.25, "squash": 0.35, "pitch": 10, "color_idle": (200, 100, 100), "sound_path": "sounds/tom_floor.wav"},
            "Ride Cymbal": {"center": (0.95, -0.80, -0.60), "draw_radius": 0.59, "hit_radius": 0.30, "squash": 0.45, "pitch": 15, "color_idle": (0, 215, 255), "sound_path": "sounds/Ride Cymbal Edge Sample.mp3"},
            "Crash Cymbal": {"center": (-1.10, -1.20, -0.92), "draw_radius": 0.47, "hit_radius": 0.25, "squash": 0.45, "pitch": 15, "color_idle": (0, 215, 255), "sound_path": "sounds/High Crash Cymbal Sample.mp3"}
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
            drum_thickness = 0.15 
            
            # IT'S A HIT ONLY IF:
            if surface_distance <= radius and depth_distance <= drum_thickness:
                if current_time - self.last_hit_time[drum_name] > self.hit_cooldown:
                    self.last_hit_time[drum_name] = current_time
                    
                    # --- NEW: Play the sound instantly! ---
                    if drum_name in self.loaded_sounds:
                        self.loaded_sounds[drum_name].play()
                        
                    return drum_name
        return None

    def cleanup(self):
        """Call this when your app closes to safely shut down the audio engine."""
        pygame.quit()