import cv2
import numpy as np
import math

class UIRenderer:
    def __init__(self, frame_w, frame_h, focal_length):
        self.frame_width = frame_w
        self.frame_height = frame_h
        self.focal_length = focal_length
        self.app_h = 950
        self.right_w = 640
        self.pov_h = 540
        self.win_name = "AR Drum Kit"

        # --- FULL SCREEN CONFIGURATION ---
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty(self.win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def draw_combined_view(self, cam_img, pov_canvas, controls_panel, rhythm_overlay_text=None):
        """Assembles the 3 panels into the final window."""
        cam_w = int(self.frame_width * (self.app_h / self.frame_height))
        total_w = cam_w + self.right_w
        
        combined = np.zeros((self.app_h, total_w, 3), dtype=np.uint8)

        # 1. Main Camera View (Left side)
        cam = cv2.resize(cam_img, (cam_w, self.app_h))
        if rhythm_overlay_text:
            cv2.putText(cam, rhythm_overlay_text, (12, self.app_h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 200), 2)
        combined[:, :cam_w] = cam
        cv2.line(combined, (cam_w, 0), (cam_w, self.app_h), (40, 40, 40), 2)

        # 2. POV Window (Top Right)
        if pov_canvas is not None:
            pov_resized = cv2.resize(pov_canvas, (self.right_w, self.pov_h))
        else:
            pov_resized = np.full((self.pov_h, self.right_w, 3), (12, 14, 22), dtype=np.uint8)
            cv2.putText(pov_resized, "POV hidden  [P] to show",
                        (self.right_w // 2 - 130, self.pov_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 80), 1)

        combined[:self.pov_h, cam_w:] = pov_resized
        cv2.line(combined, (cam_w, self.pov_h), (total_w, self.pov_h), (40, 40, 40), 2)

        # 3. Controls Panel (Bottom Right)
        if controls_panel is not None:
            ctrl_h = self.app_h - self.pov_h
            ctrl_resized = cv2.resize(controls_panel, (self.right_w, ctrl_h))
            combined[self.pov_h:, cam_w:] = ctrl_resized
        
        cv2.imshow(self.win_name, combined)

    def build_controls_panel(self, state_dict):
        """Builds the 2D controls panel using a dictionary of state variables."""
        w = self.right_w
        h = self.app_h - self.pov_h
        
        dbg_l = state_dict.get("dbg_l")
        dbg_r = state_dict.get("dbg_r")
        dbg_foot = state_dict.get("dbg_foot")
        cur_time = state_dict.get("cur_time", 0)
        
        panel = np.full((h, w, 3), (14, 16, 24), dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (w, 32), (24, 28, 42), -1)
        cv2.putText(panel, "CONTROLS", (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 210, 210), 1)

        # Distance logic
        dist_m = state_dict.get("dist_m")
        if dist_m is not None:
            dist_txt = f"Distance: {dist_m:.2f} m"
            dist_col = (0, 210, 210)
        else:
            dist_txt = "Distance: calibrating…"
            dist_col = (100, 100, 120)

        cv2.rectangle(panel, (8, 36), (w - 8, 62), (24, 32, 50), -1)
        cv2.rectangle(panel, (8, 36), (w - 8, 62), (50, 60, 90), 1)
        ts = cv2.getTextSize(dist_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        cv2.putText(panel, dist_txt, ((w - ts[0]) // 2, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, dist_col, 1)

        # Build Keyboard mappings dynamically from state
        keys = [
            ("[D]",   "Toggle drums",   state_dict.get("show_drums", True)),
            ("[N]",   "Drum names",     state_dict.get("show_drum_names", True)),
            ("[P]",   "POV window",     state_dict.get("show_pov", True)),
            ("[S]",   "Stick mode",     state_dict.get("stick_mode", False)),
            ("[C]",   "Coords overlay", state_dict.get("show_coords", False)),
            ("[F]",   "Freeze drums",   state_dict.get("freeze_drums", True)),
            ("[O]",   "Flow debug",     state_dict.get("show_flow", False)),
            ("[Q]",   state_dict.get("depth_label", "Depth"), state_dict.get("depth_state", None)),
            ("[R]",   "Rhythm test",    state_dict.get("rhythm_active", False)),
            ("[ESC]", "Quit",           None),
        ]

        DIST_BOX_BOTTOM = 68
        row_h = max(26, (h - DIST_BOX_BOTTOM - 40) // len(keys))
        for i, (key, label, state) in enumerate(keys):
            y  = DIST_BOX_BOTTOM + 12 + i * row_h
            kw = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
            cv2.rectangle(panel, (10, y - 14), (10 + kw + 10, y + 6), (34, 38, 60), -1)
            cv2.rectangle(panel, (10, y - 14), (10 + kw + 10, y + 6), (60, 65, 100), 1)
            cv2.putText(panel, key,   (15, y),           cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 255), 1)
            cv2.putText(panel, label, (10 + kw + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 180), 1)

            if state is not None:
                dot_col = (0, 220, 110) if state else (80, 80, 100)
                dot_x   = w - 22
                cv2.circle(panel, (dot_x, y - 4), 6, dot_col, -1)
                lbl = "ON" if state else "off"
                cv2.putText(panel, lbl, (dot_x - 26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, dot_col, 1)

        # Hit detection overlays
        base_y = h - 18
        col_hit = (0, 255, 120)
        if dbg_l and (cur_time - state_dict.get("last_l_hit_time", 0)) < 0.4:
            cv2.putText(panel, "LEFT  HIT!", (14, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col_hit, 2)
        if dbg_r and (cur_time - state_dict.get("last_r_hit_time", 0)) < 0.4:
            cv2.putText(panel, "RIGHT HIT!", (w // 2 - 60, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col_hit, 2)
        if dbg_foot and (cur_time - state_dict.get("last_foot_hit_time", 0)) < 0.4:
            cv2.putText(panel, "KICK!", (w // 2 - 22, base_y - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 130, 255), 2)

        depth_msg = state_dict.get("depth_status_msg", "")
        if depth_msg:
            col = (0, 200, 100) if state_dict.get("depth_active") else (120, 120, 140)
            cv2.putText(panel, depth_msg, (10, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

        return panel

    def draw_2d_overlays(self, image, dbg_l, dbg_r, dbg_foot, show_coords=False):
        """Restores the 2D tracking circles on the webcam feed."""
        for dbg, color in [(dbg_l, (255, 0, 0)), (dbg_r, (0, 0, 255))]:
            if dbg:
                px = dbg["pos_px"]
                cv2.circle(image, px, 15, (0, 255, 0) if dbg["hit"] else color, -1)
                if show_coords:
                    cv2.putText(image, f"STATE:{dbg['state']} Z:{dbg['z']:.2f}",
                                (px[0] - 40, px[1] - 40), 0, 1.2, (0, 0, 255), 2)
        
        if dbg_foot:
            px = dbg_foot["pos_px"]
            is_hit = bool(dbg_foot.get("hit"))
            is_pressing = dbg_foot.get("state") == "DOWN"
            ring_col = (0, 130, 255) if is_pressing else (80, 60, 30)
            cv2.circle(image, px, 20, ring_col, 2)
            dot_col = (0, 200, 255) if is_hit else (180, 100, 30)
            cv2.circle(image, px, 10, dot_col, -1)
            if is_hit:
                cv2.putText(image, "KICK", (px[0] - 20, px[1] - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    def draw_drums_2d(self, image, kit, cur_time, show_drum_names=True):
        """Draws the 2D drum ellipses directly onto the webcam feed."""
        if not kit.pixel_positions:
            return
            
        overlay = image.copy()
        for name, pos in kit.pixel_positions.items():
            is_hit = (
                cur_time - kit.last_hit_time["L"].get(name, 0) < 0.2 or
                cur_time - kit.last_hit_time["R"].get(name, 0) < 0.2 or
                cur_time - kit.last_hit_time["RF"].get(name, 0) < 0.2
            )
            color = (0, 255, 0) if is_hit else kit.drums[name]["color_idle"]
            cv2.ellipse(overlay, (pos["cx"], pos["cy"]),
                        (max(pos["rx"], 4), max(pos["ry"], 2)), 0, 0, 360, color, -1)
                        
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        if show_drum_names:
            for name, pos in kit.pixel_positions.items():
                label = name.upper()
                ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                tx, ty = pos["cx"] - ts[0] // 2, pos["cy"] + ts[1] // 2
                cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def render_pov_canvas(self, kit, w_lm_eff, cur_time, dbg_l, dbg_r, dbg_foot, fixed_sw_m, state_dict):
        """Builds the 3D projected perspective."""
        _POV_REN_W = 800
        _POV_REN_H = 640
        canvas = np.full((_POV_REN_H, _POV_REN_W, 3), (8, 10, 18), dtype=np.uint8)

        SCALE  = 170
        cx_c   = _POV_REN_W // 2
        cy_c   = (_POV_REN_H // 2) - 100

        CAM_X, CAM_Y, CAM_Z = 0.0, 0.85, 0.0
        PITCH  = math.radians(22)
        cos_p, sin_p = math.cos(PITCH), math.sin(PITCH)

        def project(x, y, z):
            x += CAM_X;  y += CAM_Y;  z += CAM_Z
            y_rot = y * cos_p + z * sin_p
            z_rot = z * cos_p - y * sin_p
            dist  = max(abs(z_rot), 0.1)
            px    = int(cx_c + (x / dist) * SCALE)
            py    = int(cy_c + (y_rot / dist) * SCALE)
            return px, py, z_rot, dist

        rq = []

        # Ground grid
        XS = [-1.1, -0.7, -0.28, 0, 0.28, 0.7, 1.1]
        ZS = [-0.15, -0.35, -0.55, -0.75, -0.95, -1.15]
        GRID_Y = 0.65

        for gx in XS:
            p1 = project(gx, GRID_Y, ZS[0]);  p2 = project(gx, GRID_Y, ZS[-1])
            rq.append({"t":"line", "depth": -100, "p1":p1[:2], "p2":p2[:2], "color":(30,42,58), "w":1})
        for gz in ZS:
            p1 = project(XS[0], GRID_Y, gz);  p2 = project(XS[-1], GRID_Y, gz)
            rq.append({"t":"line", "depth": -100, "p1":p1[:2], "p2":p2[:2], "color":(30,42,58), "w":1})

        # Drums (Sorted by projected camera depth Z-Rot)
        for name, props in kit.drums.items():
            cx, cy, cz = props["center"]
            is_hit = (
                cur_time - kit.last_hit_time["L"].get(name, 0) < 0.20 or
                cur_time - kit.last_hit_time["R"].get(name, 0) < 0.20 or
                cur_time - kit.last_hit_time["RF"].get(name, 0) < 0.20
            )
            col = (0, 240, 60) if is_hit else props["color_idle"]
            
            # Use z_rot for depth sorting instead of flat cz
            ppx, ppy, z_rot, dist = project(cx, cy, cz)
            rx_m, ry_m, rz_m = props["radii"]

            if name == "Bass Drum":
                visual_thickness_m = 0.20
                rx    = max(int((rx_m * SCALE) / dist), 4)
                ry    = max(int((rx_m * cos_p * SCALE) / dist), 4)
                thick = max(int((visual_thickness_m * SCALE) / dist), 2)
                rq.append({"t":"bass_drum", "depth": z_rot, "name":"BD",
                           "px":ppx, "py":ppy, "rx":rx, "ry":ry, "thick":thick, "col":col})
            else:
                visual_thickness_m = 0.02 if "Cymbal" in name or "Hi-Hat" in name else 0.12
                rx    = max(int((rx_m * SCALE) / dist), 4)
                ry    = max(int((ry_m * SCALE) / dist), 2)
                thick = max(int((visual_thickness_m * SCALE) / dist), 2)
                rq.append({"t":"drum", "depth": z_rot, "name":name[:3].upper(),
                           "px":ppx, "py":ppy, "rx":rx, "ry":ry, "thick":thick, "col":col})

        # Arms (Restored natural rigid skeleton)
        if w_lm_eff and fixed_sw_m > 0:
            arm_defs = [
                (w_lm_eff[11], w_lm_eff[13], w_lm_eff[15], w_lm_eff[19], dbg_l, (100, 220, 255)),
                (w_lm_eff[12], w_lm_eff[14], w_lm_eff[16], w_lm_eff[20], dbg_r, ( 80,  80, 255)),
            ]
            for sh_w, el_w, wr_w, fi_w, dbg, arm_col in arm_defs:
                if sh_w.visibility < 0.3 or el_w.visibility < 0.3 or wr_w.visibility < 0.3:
                    continue
                sh3 = (sh_w.x, sh_w.y, sh_w.z)
                el3 = (el_w.x, el_w.y, el_w.z)
                wr3 = (wr_w.x, wr_w.y, wr_w.z)
                fi3 = (fi_w.x, fi_w.y, fi_w.z)
                
                sh_p, el_p, wr_p = project(*sh3), project(*el3), project(*wr3)
                line_w = max(2, int(8 / el_p[3]))
                
                # Sorted cleanly using the camera's Z projection
                rq.append({"t":"line", "depth": (sh_p[2]+el_p[2])/2, "p1":sh_p[:2], "p2":el_p[:2], "color":arm_col, "w":line_w})
                rq.append({"t":"line", "depth": (el_p[2]+wr_p[2])/2, "p1":el_p[:2], "p2":wr_p[:2], "color":arm_col, "w":line_w})
                
                rad_sh = max(3, int(10 / sh_p[3]))
                rad_el = max(2, int( 8 / el_p[3]))
                rq.append({"t":"dot", "depth": sh_p[2], "px":sh_p[0], "py":sh_p[1], "col":arm_col, "r":rad_sh})
                rq.append({"t":"dot", "depth": el_p[2], "px":el_p[0], "py":el_p[1], "col":arm_col, "r":rad_el})

                # Virtual Drumsticks
                fw_x = fi3[0]-wr3[0]; fw_y = fi3[1]-wr3[1]; fw_z = fi3[2]-wr3[2]
                fw_mag = math.sqrt(fw_x**2 + fw_y**2 + fw_z**2)
                if state_dict.get("stick_mode") and fw_mag > 1e-3:
                    ext_len = state_dict.get("stick_length", 0.1)
                    tip3 = (wr3[0]+(fw_x/fw_mag)*ext_len,
                            wr3[1]+(fw_y/fw_mag)*ext_len,
                            wr3[2]+(fw_z/fw_mag)*ext_len)
                    tp_p = project(*tip3)
                    stick_w = max(1, int(6 / tp_p[3]))
                    tip_r   = max(2, int(10 / tp_p[3]))
                    
                    rq.append({"t":"line", "depth": (wr_p[2]+tp_p[2])/2, "p1":wr_p[:2], "p2":tp_p[:2], "color":(255,220,50), "w":stick_w})
                    rq.append({"t":"dot", "depth": tp_p[2], "px":tp_p[0], "py":tp_p[1], "col":(255,220,50), "r":tip_r})

                is_hit_wrist = dbg.get("hit", False) if dbg else False
                wrist_col    = (0, 255, 60) if is_hit_wrist else arm_col
                rad_wr = max(4, int(14 / wr_p[3]))
                rq.append({"t":"hand", "depth": wr_p[2], "px":wr_p[0], "py":wr_p[1],
                           "col":wrist_col, "state":dbg.get("state","") if dbg else "", "r":rad_wr})

        # Render Queue processing
        rq.sort(key=lambda i: i["depth"])
        for item in rq:
            t = item["t"]
            if t == "line":
                cv2.line(canvas, item["p1"], item["p2"], item["color"], item["w"])
            elif t == "dot":
                cv2.circle(canvas, (item["px"],item["py"]), item["r"], item["col"], -1)
            elif t == "bass_drum":
                ppx, ppy = item["px"], item["py"]
                rx, ry, thick, c = item["rx"], item["ry"], item["thick"], item["col"]
                shade  = (int(c[0]*0.4), int(c[1]*0.4), int(c[2]*0.4))
                back_y = ppy - thick
                cv2.ellipse(canvas, (ppx, back_y), (rx, ry), 0, 0, 360, shade, -1)
                pts = np.array([(ppx-rx,ppy),(ppx+rx,ppy),(ppx+rx,back_y),(ppx-rx,back_y)], dtype=np.int32)
                cv2.fillPoly(canvas, [pts], shade)
                head_col = (200, 255, 200) if c == (0, 240, 60) else (170, 170, 170)
                cv2.ellipse(canvas, (ppx, ppy), (rx, ry), 0, 0, 360, head_col, -1)
                cv2.ellipse(canvas, (ppx, ppy), (rx, ry), 0, 0, 360, c, max(3, int(rx*0.15)))
                cv2.ellipse(canvas, (ppx, ppy), (rx+2, ry+2), 0, 0, 360, (40,40,40), 1)
                lbl = item["name"]
                ts  = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.putText(canvas, lbl, (ppx-ts[0]//2, ppy+ts[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                cv2.putText(canvas, lbl, (ppx-ts[0]//2, ppy+ts[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            elif t == "drum":
                ppx, ppy = item["px"], item["py"]
                rx, ry, thick, c = item["rx"], item["ry"], item["thick"], item["col"]
                shade = (int(c[0]*.45), int(c[1]*.45), int(c[2]*.45))
                cv2.ellipse(canvas, (ppx, ppy+thick), (rx, ry), 0, 0, 360, shade, -1)
                pts = np.array([(ppx-rx,ppy),(ppx+rx,ppy),(ppx+rx,ppy+thick),(ppx-rx,ppy+thick)], dtype=np.int32)
                cv2.fillPoly(canvas, [pts], shade)
                cv2.ellipse(canvas, (ppx, ppy), (rx, ry), 0, 0, 360, c, -1)
                cv2.ellipse(canvas, (ppx, ppy), (rx, ry), 0, 0, 360, (170,170,170), 1)
                lbl = item["name"]
                ts  = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
                cv2.putText(canvas, lbl, (ppx-ts[0]//2, ppy+ts[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0), 3)
                cv2.putText(canvas, lbl, (ppx-ts[0]//2, ppy+ts[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)
            elif t == "hand":
                ppx, ppy, r, c = item["px"], item["py"], item["r"], item["col"]
                if item["state"] == "DOWN": cv2.circle(canvas, (ppx,ppy), r+8, c, 2)
                cv2.circle(canvas, (ppx,ppy), r, c, -1)
                cv2.circle(canvas, (ppx,ppy), r, (255,255,255), 1)

        if state_dict.get("depth_active"):
            cv2.putText(canvas, "DEPTH-ANYTHING ON", (12, _POV_REN_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 120), 1)

        cv2.putText(canvas, "TRUE 1ST PERSON POV", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 210), 1)

        if dbg_foot is not None:
            foot_label = f"KICK: {dbg_foot.get('state','UP')}"
            foot_col   = (0, 200, 255) if dbg_foot.get("state") == "DOWN" else (80, 80, 100)
            cv2.putText(canvas, foot_label, (_POV_REN_W - 160, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, foot_col, 1)

        return canvas