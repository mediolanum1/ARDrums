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
            "Snare": {"center": (0.0, -0.2, -0.3),"dist":(0.5,0.3,0.2), "squash": 0.35, "pitch": 10, "color_idle": (200, 200, 200), "sound_path": "sounds/Snare Sample.mp3"},
           # "Bass Drum": {"center": (0.35, 0.40, -0.55), "draw_radius": 0.65, "hit_radius": 0.35, "squash": 0.85, "pitch": 0, "color_idle": (50, 50, 50), "sound_path": "sounds/kick.wav"},
            "Hi-Hat": {"center": (-1.20, -0.65, -0.45), "dist":(0.5,0.3,0.3), "squash": 0.30, "pitch": 5, "color_idle": (0, 200, 255), "sound_path": "sounds/HI-HAT Top Sample.mp3"},
            "High Tom": {"center": (-0.30, -0.6, -0.70), "dist":(0.3,0.3,0.2),  "squash": 0.60, "pitch": 25, "color_idle": (255, 100, 100), "sound_path": "sounds/High Tom Sample.mp3"},
            "Mid Tom": {"center": (0.3, -0.6, -0.70), "dist":(0.3,0.3,0.2),  "squash": 0.60, "pitch": 25, "color_idle": (255, 100, 100), "sound_path": "sounds/Middle Tom Sample.mp3"},
            #"Floor Tom": {"center": (1.10, -0.20, -0.50), "draw_radius": 0.47, "hit_radius": 0.25, "squash": 0.35, "pitch": 10, "color_idle": (200, 100, 100), "sound_path": "sounds/tom_floor.wav"},
            "Ride Cymbal": {"center": (1.10, -1.20, -0.50), "dist":(0.3,0.3,0.3), "squash": 0.45, "pitch": 15, "color_idle": (0, 215, 255), "sound_path": "sounds/Ride Cymbal Edge Sample.mp3"},
            "Crash Cymbal": {"center": (0.20, -0.85, -1.10), "dist":(0.3,0.3,0.2),  "squash": 0.45, "pitch": 15, "color_idle": (0, 215, 255), "sound_path": "sounds/High Crash Cymbal Sample.mp3"}
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

    
    def check_line_intersection(self, p, cur_time):
        """
        Check if the line segment from p1 to p2 intersects any drum in 2D (X-Y) after filtering by Z range.
        p1: (x, y, z) from the previous frame
        p2: (x, y, z) from the current frame
        """
        x, y, z = p
        # Step 1: Filter possible drums based on Z range of p2
        possible_drums = []
        for drum_name, props in self.drums.items():
            cx, cy, cz = props["center"]
            thickness = props["dist"][2]  # The vertical thickness of the drum
            drum_top_z = cz - (thickness / 2)
            drum_bottom_z = cz + (thickness / 2)
            if drum_top_z <= z <= drum_bottom_z:
                possible_drums.append(drum_name)
        
        # Step 2: For each possible drum, check 2D intersection in X-Y plane
        for drum_name in possible_drums:
            props = self.drums[drum_name]
            cx, cy, cz = props["center"]
            dist_x,dist_y,dist_z= props["dist"]
            
            if (x-cx)**2/dist_x**2+(y-cy)**2/dist_y**2 <=1:
                if cur_time - self.last_hit_time[drum_name] > self.hit_cooldown:
                        self.last_hit_time[drum_name] = cur_time
                        if drum_name in self.loaded_sounds:
                            self.loaded_sounds[drum_name].play()
                        return drum_name
            
            
                    
        
        return None

    def cleanup(self):
        """Call this when your app closes to safely shut down the audio engine."""
        pygame.quit()