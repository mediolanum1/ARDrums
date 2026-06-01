import math
import os
import pygame

class VirtualDrumKit:
    def __init__(self, base_dir="ARDrum_kit/drums"):
        """
        Manages the 3D layout, 2D pixel projections, and dual-layered audio 
        playback for the AR Drum Kit.
        """
        self.hit_cooldown     = 0.25
        self.use_sticks       = False
        self.active_stick_ext = (0.0, 0.0, 0.0)
        self.pixel_positions  = {}
        
        # Track hit times per limb (Left, Right, Right Foot)
        self.last_hit_time = {"L": {}, "R": {}, "RF": {}}
      
        # Velocity thresholds based on 'smooth_norm_speed'
        self.MIN_SPEED = 0.015  # Softest hit
        self.MAX_SPEED = 0.15   # Hardest hit
        
        try:
            pygame.mixer.pre_init(...)
            pygame.init()
        except Exception as e:
            print(f"[AUDIO] Pygame mixer failed...")

  
        self.drums = {
            "Snare": {
                "center": ( 0.00, -0.10, -0.2), "radii": (0.22, 0.1, 0.12), "color_idle": (200, 200, 200), 
                "sound_path": f"{base_dir}/sounds/snare.mp3", 
                "sound_path_quiet": f"{base_dir}/sounds/snare_quiet.mp3"
            },
            "Hi-Hat": {
                "center": (-0.44, -0.28, -0.2), "radii": (0.15, 0.06, 0.12), "color_idle": (0,   200, 255), 
                "sound_path": f"{base_dir}/sounds/hi_hat.mp3", 
                "sound_path_quiet": f"{base_dir}/sounds/hi_hat_quiet.mp3"
            },
            "High Tom": {
                "center": (-0.15, -0.35, -0.33), "radii": (0.14, 0.06, 0.1), "color_idle": (255, 100, 100), 
                "sound_path": f"{base_dir}/sounds/high_tom.mp3", 
                "sound_path_quiet": f"{base_dir}/sounds/high_tom_quiet.mp3"
            },
            "Mid Tom": {
                "center": ( 0.15, -0.35, -0.33), "radii": (0.14, 0.06, 0.1), "color_idle": (255, 100, 100), 
                "sound_path": f"{base_dir}/sounds/middle_tom.mp3", 
                "sound_path_quiet": f"{base_dir}/sounds/middle_tom_quiet.mp3"
            },
            "Ride Cymbal": {
                "center": ( 0.55, -0.40, -0.33), "radii": (0.20, 0.07, 0.12), "color_idle": (0,   215, 255), 
                "sound_path": f"{base_dir}/sounds/ride_cymbal.mp3", 
                "sound_path_quiet": f"{base_dir}/sounds/ride_cymbal_quiet.mp3"
            },
            "Crash Cymbal": {
                "center": (-0.30, -0.60, -0.36), "radii": (0.20, 0.07, 0.1), "color_idle": (0,   215, 255), 
                "sound_path": f"{base_dir}/sounds/high_crash_cymbal.mp3", 
                "sound_path_quiet": f"{base_dir}/sounds/high_crash_cymbal_quiet.mp3"
            },
            "Bass Drum": {
                "center": ( 0.10,  0.1, -0.25), "radii": (0.22, 0.22, 0.10), "color_idle": (220,  90,  30), 
                "sound_path": f"{base_dir}/sounds/kick.mp3", 
                "sound_path_quiet": f"{base_dir}/sounds/kick_quiet.mp3"
            },
        }

        self._init_audio()

    def _init_audio(self):
        """Initializes Pygame mixer and loads all audio layers into memory."""
        try:
            pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.init()
            pygame.mixer.set_num_channels(64)
        except Exception as e:
            print(f"[AUDIO] Pygame mixer failed to init: {e}")

        self.loaded_sounds = {}
        self.loaded_sounds_quiet = {}
        
        for drum_name, props in self.drums.items():
            self.last_hit_time["L"][drum_name] = 0.0
            self.last_hit_time["R"][drum_name] = 0.0
            self.last_hit_time["RF"][drum_name] = 0.0
            
            # Normal Samples
            if os.path.exists(props["sound_path"]):
                self.loaded_sounds[drum_name] = pygame.mixer.Sound(props["sound_path"])
            else:
                print(f"[AUDIO] WARNING: Missing normal audio -> {props['sound_path']}")
                
            # Quiet Samples
            if os.path.exists(props["sound_path_quiet"]):
                self.loaded_sounds_quiet[drum_name] = pygame.mixer.Sound(props["sound_path_quiet"])
            else:
                print(f"[AUDIO] WARNING: Missing quiet audio -> {props['sound_path_quiet']}")

    # ─── Dynamic Layout Updater (NEW) ──────────────────────────────────────────
    def update_layout(self, torso_cx, torso_cy, scale, ankle_pos=None):
        """
        Calculates the 2D pixel coordinates for all drums based on the user's 
        current body position on screen.
        
        :param torso_cx: Center X of the user's hips (pixels)
        :param torso_cy: Center Y of the user's hips (pixels)
        :param scale: The metric_to_px_scale from calibration
        :param ankle_pos: Tuple (x, y) for the bass drum anchor, if visible.
        """
        positions = {}
        
        for name, props in self.drums.items():
            if name == "Bass Drum":
                if ankle_pos is not None:
                    rx_m, ry_m, _ = props["radii"]
                    positions[name] = {
                        "cx": int(ankle_pos[0]),
                        "cy": int(ankle_pos[1]),
                        "rx": int(rx_m * scale * 0.9),
                        "ry": int(ry_m * scale * 0.45),
                    }
                continue

            cx_m, cy_m, _ = props["center"]
            rx_m, ry_m, _ = props["radii"]
            
            positions[name] = {
                "cx": int(torso_cx + cx_m * scale),
                "cy": int(torso_cy + cy_m * scale),
                "rx": int(rx_m * scale),
                "ry": int(ry_m * scale),
            }

        self.pixel_positions = positions

    # ─── Velocity & Triggers ──────────────────────────────────────────────────
    def get_hit_parameters(self, smooth_norm_speed: float) -> tuple[bool, float]:
        """Evaluates speed and returns: (use_quiet_sample, volume)"""
        raw_velocity = (smooth_norm_speed - self.MIN_SPEED) / (self.MAX_SPEED - self.MIN_SPEED)
        clamped_velocity = max(0.0, min(1.0, raw_velocity))
        
        if clamped_velocity < 0.6:
            quiet_normalized = clamped_velocity / 0.6
            perceived_volume = quiet_normalized ** 2.0 
            return True, max(0.1, perceived_volume)
        else:
            return False, 1.0

    def trigger_bass_drum(self, cur_time: float, smooth_norm_speed: float, hand_id: str = "RF") -> str | None:
        """Called directly by the Foot Processor to trigger the kick."""
        drum_name = "Bass Drum"
        if cur_time - self.last_hit_time[hand_id].get(drum_name, 0.0) <= self.hit_cooldown:
            return None

        self.last_hit_time[hand_id][drum_name] = cur_time

        is_quiet, volume = self.get_hit_parameters(smooth_norm_speed)
        target_dict = self.loaded_sounds_quiet if is_quiet else self.loaded_sounds

        if drum_name in target_dict:
            sound = target_dict[drum_name]
            sound.set_volume(volume)
            sound.play()

        return drum_name

    # ─── Math & Collisions ────────────────────────────────────────────────────
    @staticmethod
    def _segment_hits_ellipse(px0, py0, px1, py1, cx, cy, rx, ry):
        dx = px1 - px0; dy = py1 - py0
        fx = px0 - cx;  fy = py0 - cy

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
        """Called by the Wrist Processors to see if the gesture intersected a drum."""
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

            if (z0 < cz - rz and z1 < cz - rz) or (z0 > cz + rz and z1 > cz + rz):
                continue

            cx = pos["cx"];  cy = pos["cy"]
            rx = pos["rx"];  ry = pos["ry"]

            if rx <= 0 or ry <= 0:
                continue

            if self._segment_hits_ellipse(
                px_prev[0], px_prev[1], px_curr[0], px_curr[1], cx, cy, rx, ry
            ):
                self.last_hit_time[hand_id][drum_name] = cur_time
                
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