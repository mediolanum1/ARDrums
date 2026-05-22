import math
import os
import pygame

class VirtualDrumKit:
    def __init__(self):
        self.last_hit_time = {}
        self.hit_cooldown = 0.2

        # --- Global cooldown: after ANY drum is hit, ALL drums are locked ---
   
  
        # ── Stick mode ────────────────────────────────────────────────────────
        self.use_sticks       = False
        self.active_stick_ext = (0.0, 0.0, 0.0)

        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.init()
        pygame.mixer.set_num_channels(16)

        # ── Kit layout in TRUE METERS ─────────────────────────────────────────
        # Origin (0,0,0) : Midpoint between your hips
        # X : Left (-) / Right (+)
        # Y : Up (-) / Down (+)  [Shoulders are roughly Y = -0.5m]
        # Z : Forward (-) / Back (+) [Fully extended arm is ~ -0.6m]
        # 
        # "radii": (X_radius_m, Y_radius_m, Z_half_thickness_m)
        # This replaces the old "squash" hack. Visuals and hitboxes are now 1:1.
        self.drums = {
            "Snare":        {"center": (0.0,  -0.15, -0.3), "radii": (0.16, 0.07, 0.5), "color_idle": (200, 200, 200), "sound_path": "sounds/Snare Sample.mp3"},
           # "Hi-Hat":       {"center": (-0.35, -0.25, -0.3), "radii": (0.15, 0.06, 0.08), "color_idle": (0, 200, 255), "sound_path": "sounds/HI-HAT Top Sample.mp3"},
           ## "High Tom":     {"center": (-0.15, -0.35, -0.35), "radii": (0.14, 0.06, 0.12), "color_idle": (255, 100, 100), "sound_path": "sounds/High Tom Sample.mp3"},
            #"Mid Tom":      {"center": (0.15,  -0.35, -0.35), "radii": (0.14, 0.06, 0.12), "color_idle": (255, 100, 100), "sound_path": "sounds/Middle Tom Sample.mp3"},
            #"Ride Cymbal":  {"center": (0.45,  -0.30, -0.3), "radii": (0.20, 0.07, 0.08), "color_idle": (0, 215, 255), "sound_path": "sounds/Ride Cymbal Edge Sample.mp3"},
            #"Crash Cymbal": {"center": (-0.25, -0.55, -0.37), "radii": (0.20, 0.07, 0.08), "color_idle": (0, 215, 255), "sound_path": "sounds/High Crash Cymbal Sample.mp3"}
        }

        self.loaded_sounds = {}
        for drum_name, props in self.drums.items():
            self.last_hit_time[drum_name] = 0.0
            path = props["sound_path"]
            if os.path.exists(path):
                self.loaded_sounds[drum_name] = pygame.mixer.Sound(path)
            else:
                print(f"WARNING: Missing audio file -> {path}")

    def check_line_intersection(self, p_prev, p_curr, cur_time, smooth_norm_speed):
        for drum_name, props in self.drums.items():
            cz = props["center"][2]
            rz = props["radii"][2]
            z0, z1 = p_prev[2], p_curr[2]
            # Did the segment cross the Z slab?
            if not ((z0 <= cz + rz and z1 >= cz - rz) or 
                    (z1 <= cz + rz and z0 >= cz - rz)):
                continue
            # Interpolate position at the crossing point
            t = (cz - z0) / (z1 - z0) if z1 != z0 else 0.5
            xi = p_prev[0] + t * (p_curr[0] - p_prev[0])
            yi = p_prev[1] + t * (p_curr[1] - p_prev[1])
            cx, cy, _ = props["center"]
            rx, ry, _ = props["radii"]
            if (xi - cx)**2 / rx**2 + (yi - cy)**2 / ry**2 <= 1:
                # cooldown check + play sound
                if cur_time - self.last_hit_time[drum_name] > self.hit_cooldown:
                    self.last_hit_time[drum_name] = cur_time
                  
                    if drum_name in self.loaded_sounds: 
                        sound = self.loaded_sounds[drum_name]
                        sound.set_volume(min(1.0, smooth_norm_speed * 8))
                        sound.play()
                    return drum_name

        return None

    def cleanup(self):
        pygame.quit()