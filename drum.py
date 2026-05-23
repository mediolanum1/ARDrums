import math
import os
import pygame

class VirtualDrumKit:
    def __init__(self):
  
        self.hit_cooldown     = 0.2
        self.use_sticks       = False
        self.active_stick_ext = (0.0, 0.0, 0.0)
        self.pixel_positions  = {}
        self.last_hit_time = {"L": {}, "R": {}, "RF":{}}
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.init()
        pygame.mixer.set_num_channels(16)

        self.drums = {
            "Snare":        {"center": ( 0.00, -0.10, -0.33), "radii": (0.16, 0.07, 0.08), "color_idle": (200, 200, 200), "sound_path": "sounds/Snare Sample.mp3"},
            "Hi-Hat":       {"center": (-0.44, -0.28, -0.32), "radii": (0.15, 0.06, 0.06), "color_idle": (0,   200, 255), "sound_path": "sounds/HI-HAT Top Sample.mp3"},
            "High Tom":     {"center": (-0.15, -0.35, -0.49), "radii": (0.14, 0.06, 0.06), "color_idle": (255, 100, 100), "sound_path": "sounds/High Tom Sample.mp3"},
            "Mid Tom":      {"center": ( 0.15, -0.35, -0.49), "radii": (0.14, 0.06, 0.06), "color_idle": (255, 100, 100), "sound_path": "sounds/Middle Tom Sample.mp3"},
            "Ride Cymbal":  {"center": ( 0.55, -0.40, -0.38), "radii": (0.20, 0.07, 0.08), "color_idle": (0,   215, 255), "sound_path": "sounds/Ride Cymbal Edge Sample.mp3"},
            "Crash Cymbal": {"center": (-0.30, -0.60, -0.45), "radii": (0.20, 0.07, 0.08), "color_idle": (0,   215, 255), "sound_path": "sounds/High Crash Cymbal Sample.mp3"},
            # ── Bass Drum ────────────────────────────────────────────────────
            # Positioned at floor level in front of the player.
            # World-space origin is the hip midpoint; Y positive = downward.
            # x= 0.10  : slightly right of centre (right-foot pedal)
            # y= 0.85  : roughly at ankle height below hip origin
            # z=-0.38  : in front of the player (same depth band as snare)
            # radii    : large circular shell — (rx, ry, rz_depth_slab)
            "Bass Drum":    {"center": ( 0.10,  0.1, -0.25), "radii": (0.22, 0.22, 0.10), "color_idle": (220,  90,  30), "sound_path": "sounds/Kickdrum Sample.mp3"},
        }

        self.loaded_sounds = {}
        for drum_name, props in self.drums.items():
            self.last_hit_time["L"][drum_name] = 0.0
            self.last_hit_time["R"][drum_name] = 0.0
            path = props["sound_path"]
            if os.path.exists(path):
                self.loaded_sounds[drum_name] = pygame.mixer.Sound(path)
            else:
                print(f"WARNING: Missing audio file -> {path}")

    # ── Bass-drum trigger (called by GestureFootProcessor) ────────────────────
    def trigger_bass_drum(self, cur_time: float, smooth_norm_speed: float,
                          hand_id: str = "RF") -> str | None:
        """Register a bass-drum hit originating from *hand_id* foot.

        Returns "Bass Drum" on success, None if still in cooldown.
        Uses the same cooldown logic as check_line_intersection so that
        the last_hit_time dict stays consistent for the POV renderer.
        """
        drum_name = "Bass Drum"
        if cur_time - self.last_hit_time[hand_id].get(drum_name, 0.0) <= self.hit_cooldown:
            return None

        self.last_hit_time[hand_id][drum_name] = cur_time

        if drum_name in self.loaded_sounds:
            sound = self.loaded_sounds[drum_name]
            # Bass drum volume: speed multiplier is larger (foot swings faster
            # in world-normalised terms than a wrist strike)
            sound.set_volume(min(1.0, smooth_norm_speed * 10))
            sound.play()

        return drum_name

    # ── Existing percussion trigger (wrists) ─────────────────────────────────

    @staticmethod
    def _segment_hits_ellipse(px0, py0, px1, py1, cx, cy, rx, ry):
        """
        Returns True if the line segment (px0,py0)->(px1,py1)
        intersects or is contained within the ellipse.

        Substitutes the parametric line x(t) = px0 + t*dx,
        y(t) = py0 + t*dy into the ellipse equation, producing
        a quadratic in t. The segment hits the ellipse when the
        [t1, t2] solution interval overlaps [0, 1].
        """
        dx = px1 - px0
        dy = py1 - py0
        fx = px0 - cx
        fy = py0 - cy

        a = (dx / rx) ** 2 + (dy / ry) ** 2

        if a < 1e-10:
            # Degenerate: no movement between frames — point test
            return (fx / rx) ** 2 + (fy / ry) ** 2 <= 1

        b    = 2 * (fx * dx / rx ** 2 + fy * dy / ry ** 2)
        c    = (fx / rx) ** 2 + (fy / ry) ** 2 - 1
        disc = b ** 2 - 4 * a * c

        if disc < 0:
            return False  # line misses ellipse entirely

        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2 * a)  # entry
        t2 = (-b + sqrt_disc) / (2 * a)  # exit

        # Segment overlaps ellipse when [t1, t2] overlaps [0, 1]
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
            # Bass drum is triggered separately by GestureFootProcessor
            if drum_name == "Bass Drum":
                continue

            if cur_time - self.last_hit_time[hand_id][drum_name] <= self.hit_cooldown:
                continue

            pos = self.pixel_positions.get(drum_name)
            if pos is None:
                continue

            cz = props["center"][2]
            rz = props["radii"][2]

            # Z slab gate — same as before
            if z0 < cz - rz and z1 < cz - rz:
                continue
            if z0 > cz + rz and z1 > cz + rz:
                continue

            cx = pos["cx"];  cy = pos["cy"]
            rx = pos["rx"];  ry = pos["ry"]

            if rx <= 0 or ry <= 0:
                continue

            # 2D line segment vs ellipse — catches grazing strikes too
            if self._segment_hits_ellipse(
                px_prev[0], px_prev[1],
                px_curr[0], px_curr[1],
                cx, cy, rx, ry
            ):
                self.last_hit_time[hand_id][drum_name] = cur_time
                if drum_name in self.loaded_sounds:
                    sound = self.loaded_sounds[drum_name]
                    sound.set_volume(min(1.0, smooth_norm_speed * 8))
                    sound.play()
                return drum_name

        return None

    def cleanup(self):
        pygame.quit()