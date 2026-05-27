import math
import os
import pygame


class VirtualDrumKit:
    def __init__(self):
  
        self.hit_cooldown     = 0.25
        self.use_sticks       = False
        self.active_stick_ext = (0.0, 0.0, 0.0)
        self.pixel_positions  = {}
        self.last_hit_time = {"L": {}, "R": {}, "RF":{}}
      
        # Define these thresholds based on your typical 'smooth_norm_speed' values
        self.MIN_SPEED = 0.015  # Softest hit
        self.MAX_SPEED = 0.12   # Hardest hit

        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.init()
        pygame.mixer.set_num_channels(64)

        self.drums = {
            "Snare": {
                "center": ( 0.00, -0.10, -0.36), "radii": (0.16, 0.07, 0.08), "color_idle": (200, 200, 200), 
                "sound_path": "sounds/snare.mp3", 
                "sound_path_quiet": "sounds/snare_quiet.mp3"
            },
            "Hi-Hat": {
                "center": (-0.44, -0.28, -0.36), "radii": (0.15, 0.06, 0.08), "color_idle": (0,   200, 255), 
                "sound_path": "sounds/hi_hat.mp3", 
                "sound_path_quiet": "sounds/hi_hat_quiet.mp3"
            },
            "High Tom": {
                "center": (-0.15, -0.35, -0.39), "radii": (0.14, 0.06, 0.06), "color_idle": (255, 100, 100), 
                "sound_path": "sounds/high_tom.mp3", 
                "sound_path_quiet": "sounds/high_tom_quiet.mp3"
            },
            "Mid Tom": {
                "center": ( 0.15, -0.35, -0.39), "radii": (0.14, 0.06, 0.06), "color_idle": (255, 100, 100), 
                "sound_path": "sounds/middle_tom.mp3", 
                "sound_path_quiet": "sounds/middle_tom_quiet.mp3"
            },
            "Ride Cymbal": {
                "center": ( 0.55, -0.40, -0.38), "radii": (0.20, 0.07, 0.08), "color_idle": (0,   215, 255), 
                "sound_path": "sounds/ride_cymbal.mp3", 
                "sound_path_quiet": "sounds/ride_cymbal_quiet.mp3"
            },
            "Crash Cymbal": {
                "center": (-0.30, -0.60, -0.42), "radii": (0.20, 0.07, 0.08), "color_idle": (0,   215, 255), 
                "sound_path": "sounds/high_crash_cymbal.mp3", 
                "sound_path_quiet": "sounds/high_crash_cymbal_quiet.mp3"
            },
            "Bass Drum": {
                "center": ( 0.10,  0.1, -0.25), "radii": (0.22, 0.22, 0.10), "color_idle": (220,  90,  30), 
                "sound_path": "sounds/kick.mp3", 
                "sound_path_quiet": "sounds/kick_quiet.mp3"
            },
        }

        # ── Dual Audio Loader ─────────────────────────────────────────────
        self.loaded_sounds = {}
        self.loaded_sounds_quiet = {}
        
        for drum_name, props in self.drums.items():
            self.last_hit_time["L"][drum_name] = 0.0
            self.last_hit_time["R"][drum_name] = 0.0
            self.last_hit_time["RF"][drum_name] = 0.0  # Explicitly add Right Foot
            
            path_normal = props["sound_path"]
            path_quiet = props["sound_path_quiet"]
            
            # Load Normal Samples
            if os.path.exists(path_normal):
                self.loaded_sounds[drum_name] = pygame.mixer.Sound(path_normal)
            else:
                print(f"WARNING: Missing normal audio file -> {path_normal}")
                
            # Load Quiet Samples
            if os.path.exists(path_quiet):
                self.loaded_sounds_quiet[drum_name] = pygame.mixer.Sound(path_quiet)
            else:
                print(f"WARNING: Missing quiet audio file -> {path_quiet}")


    # ── Velocity Layer Logic ──────────────────────────────────────────────────
    def get_hit_parameters(self, smooth_norm_speed: float) -> tuple[bool, float]:
        """
        Evaluates the speed and returns: (use_quiet_sample: bool, volume: float)
        """
        # Normalize the speed to a 0.0 -> 1.0 range
        raw_velocity = (smooth_norm_speed - self.MIN_SPEED) / (self.MAX_SPEED - self.MIN_SPEED)
        clamped_velocity = max(0.0, min(1.0, raw_velocity))
        
        if clamped_velocity < 0.6:
            # Map the 0.0 -> 0.6 speed range to a 0.0 -> 1.0 volume range
            # This ensures the quiet sample reaches full volume right before transitioning
            quiet_normalized = clamped_velocity / 0.6
            
            # Apply an exponential curve so very light taps are appropriately quiet
            perceived_volume = quiet_normalized ** 2.0 
            return True, max(0.1, perceived_volume)
        else:
            # Hard hit: Use normal sample at full volume
            return False, 1.0


    # ── Bass-drum trigger (called by GestureFootProcessor) ────────────────────
    def trigger_bass_drum(self, cur_time: float, smooth_norm_speed: float,
                          hand_id: str = "RF") -> str | None:
        
        drum_name = "Bass Drum"
        if cur_time - self.last_hit_time[hand_id].get(drum_name, 0.0) <= self.hit_cooldown:
            return None

        self.last_hit_time[hand_id][drum_name] = cur_time

        # Get volume and sample selection
        is_quiet, volume = self.get_hit_parameters(smooth_norm_speed)
        target_dict = self.loaded_sounds_quiet if is_quiet else self.loaded_sounds

        if drum_name in target_dict:
            sound = target_dict[drum_name]
            sound.set_volume(volume)
            sound.play()

        return drum_name


    @staticmethod
    def _segment_hits_ellipse(px0, py0, px1, py1, cx, cy, rx, ry):
        """
        Returns True if the line segment (px0,py0)->(px1,py1)
        intersects or is contained within the ellipse.
        """
        dx = px1 - px0
        dy = py1 - py0
        fx = px0 - cx
        fy = py0 - cy

        a = (dx / rx) ** 2 + (dy / ry) ** 2

        if a < 1e-10:
            return (fx / rx) ** 2 + (fy / ry) ** 2 <= 1

        b    = 2 * (fx * dx / rx ** 2 + fy * dy / ry ** 2)
        c    = (fx / rx) ** 2 + (fy / ry) ** 2 - 1
        disc = b ** 2 - 4 * a * c

        if disc < 0:
            return False 

        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2 * a)  
        t2 = (-b + sqrt_disc) / (2 * a)  

        return t1 <= 1.0 and t2 >= 0.0


    def check_line_intersection(self, p_prev, p_curr, cur_time, smooth_norm_speed,
                                 px_prev, px_curr, hand_id="R"):
        x0, y0, z0 = p_prev
        x1, y1, z1 = p_curr

        if self.use_sticks:
            ex, ey, ez = self.active_stick_ext
            x0 += ex; y0 += ey; z0 += ez
            x1 += ex; y1 += ey; z1 += ez

        for drum_name, props in self.drums.items():
            if drum_name == "Bass Drum":
                continue

            if cur_time - self.last_hit_time[hand_id][drum_name] <= self.hit_cooldown:
                continue

            pos = self.pixel_positions.get(drum_name)
            if pos is None:
                continue

            cz = props["center"][2]
            rz = props["radii"][2]

            if z0 < cz - rz and z1 < cz - rz:
                continue
            if z0 > cz + rz and z1 > cz + rz:
                continue

            cx = pos["cx"];  cy = pos["cy"]
            rx = pos["rx"];  ry = pos["ry"]

            if rx <= 0 or ry <= 0:
                continue

            if self._segment_hits_ellipse(
                px_prev[0], px_prev[1],
                px_curr[0], px_curr[1],
                cx, cy, rx, ry
            ):
                self.last_hit_time[hand_id][drum_name] = cur_time
                
                # Get volume and sample selection
                is_quiet, volume = self.get_hit_parameters(smooth_norm_speed)
                target_dict = self.loaded_sounds_quiet if is_quiet else self.loaded_sounds

                if drum_name in target_dict:
                    sound = target_dict[drum_name]
                    sound.set_volume(volume)
                    sound.play()
                    
                return drum_name

        return None

    def cleanup(self):
        pygame.quit()