import math
import os
import pygame


class VirtualDrumKit:
    def __init__(self):
        self.z_memory = {"L": 0.0, "R": 0.0}
        self.z_offset = {"L": 0.0, "R": 0.0}
        self.last_hit_time = {}
        self.hit_cooldown = 0.2

        # --- Global cooldown: after ANY drum is hit, ALL drums are locked ---
        self.global_last_hit_time = 0.0
        self.global_hit_cooldown  = 0.2

        # ── Stick mode ────────────────────────────────────────────────────────
        self.use_sticks       = False
        self.stick_length     = 0.35
        self.active_stick_ext = (0.0, 0.0, 0.0)

        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.init()
        pygame.mixer.set_num_channels(16)

        # ── Kit layout ────────────────────────────────────────────────────────
        # Coordinate origin  : hip midpoint
        # Units              : shoulder widths
        # x  left(−) / right(+)  (from player's perspective)
        # y  up(−)   / down(+)
        # z  forward(−) / back(+)  (negative = in front of player)
        #
        # z-depth changes vs previous:
        #   Snare       −1.00 → −1.20  (further in front, more natural reach)
        #   Mid Tom     −1.00 → −0.88  (slightly closer than before)
        #   Ride Cymbal −0.88 → −0.72  (closer; no z-overlap with Mid Tom:
        #                               Ride z-slab [−0.845,−0.595],
        #                               Mid Tom z-slab [−0.99,−0.77] → clear gap)
                # --- UPGRADED: Added 'sound_path' to each drum ---
        self.drums = {
            "Snare": {"center": (0.0, -0.2, -0.3),"dist":(0.5,0.3,0.2), "squash": 0.35, "pitch": 10, "color_idle": (200, 200, 200), "sound_path": "sounds/Snare Sample.mp3"},
           # "Bass Drum": {"center": (0.35, 0.40, -0.55), "draw_radius": 0.65, "hit_radius": 0.35, "squash": 0.85, "pitch": 0, "color_idle": (50, 50, 50), "sound_path": "sounds/kick.wav"},
            "Hi-Hat": {"center": (-0.9, -0.65, -0.45), "dist":(0.5,0.3,0.3), "squash": 0.30, "pitch": 5, "color_idle": (0, 200, 255), "sound_path": "sounds/HI-HAT Top Sample.mp3"},
            "High Tom": {"center": (-0.30, -0.6, -0.70), "dist":(0.3,0.3,0.2),  "squash": 0.60, "pitch": 25, "color_idle": (255, 100, 100), "sound_path": "sounds/High Tom Sample.mp3"},
            "Mid Tom": {"center": (0.3, -0.6, -0.70), "dist":(0.3,0.3,0.2),  "squash": 0.60, "pitch": 25, "color_idle": (255, 100, 100), "sound_path": "sounds/Middle Tom Sample.mp3"},
            #"Floor Tom": {"center": (1.10, -0.20, -0.50), "draw_radius": 0.47, "hit_radius": 0.25, "squash": 0.35, "pitch": 10, "color_idle": (200, 100, 100), "sound_path": "sounds/tom_floor.wav"},
            "Ride Cymbal": {"center": (0.65, -0.8, -0.50), "dist":(0.3,0.3,0.3), "squash": 0.45, "pitch": 15, "color_idle": (0, 215, 255), "sound_path": "sounds/Ride Cymbal Edge Sample.mp3"},
            "Crash Cymbal": {"center": (0.20, -0.85, -1.10), "dist":(0.3,0.3,0.2),  "squash": 0.45, "pitch": 15, "color_idle": (0, 215, 255), "sound_path": "sounds/High Crash Cymbal Sample.mp3"}
        }

        self.loaded_sounds = {}
        for drum_name, props in self.drums.items():
            self.last_hit_time[drum_name] = 0.0
            path = props["sound_path"]
            if os.path.exists(path):
                self.loaded_sounds[drum_name] = pygame.mixer.Sound(path)
            else:
                print(f"WARNING: Missing audio file -> {path}")

    # ─────────────────────────────────────────────────────────────────────────

    def check_line_intersection(self, p, cur_time):
        """
        Check whether the 3-D point p (normalised shoulder-width units, origin
        at hip midpoint) falls inside any drum zone.
        """
        x, y, z = p

        if self.use_sticks:
            ex, ey, ez = self.active_stick_ext
            x += ex
            y += ey
            z += ez

        if cur_time - self.global_last_hit_time < self.global_hit_cooldown:
            return None

        possible_drums = []
        for drum_name, props in self.drums.items():
            cx, cy, cz = props["center"]
            half_t = props["dist"][2] / 2
            if cz - half_t <= z <= cz + half_t:
                possible_drums.append(drum_name)

        for drum_name in possible_drums:
            props  = self.drums[drum_name]
            cx, cy, _ = props["center"]
            dx, dy    = props["dist"][0], props["dist"][1]
            if (x - cx) ** 2 / dx ** 2 + (y - cy) ** 2 / dy ** 2 <= 1:
                if cur_time - self.last_hit_time[drum_name] > self.hit_cooldown:
                    self.last_hit_time[drum_name] = cur_time
                    self.global_last_hit_time     = cur_time
                    if drum_name in self.loaded_sounds:
                        self.loaded_sounds[drum_name].play()
                    return drum_name

        return None

    def cleanup(self):
        pygame.quit()