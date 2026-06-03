import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
import threading
import queue
import time
import cv2
import numpy as np
from ARDrum_kit.utils.preprocessing import Preprocessor

class PoseTracker:
    def __init__(self, input_frame_queue):
        self.input_queue = input_frame_queue
        
        # holds the output (Frames + Landmarks) for the Main loop
        self.result_queue = queue.Queue(maxsize=2)
        self.running = False
        #self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.preprocessor = Preprocessor()

    def start(self):
        self.running = True
        threading.Thread(target=self._ai_thread, daemon=True).start()

    def _ai_thread(self):
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="./pose_landmarker_models/pose_landmarker_full.task"),
            running_mode=vision.RunningMode.VIDEO,
        )
        with PoseLandmarker.create_from_options(options) as pose_landmarker:
            while self.running:
                raw_image = self.input_queue.get() 
                processed_image = self.preprocessor._preprocess(raw_image)
                rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                ts_ms = int(time.time() * 1000)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                
                result = pose_landmarker.detect_for_video(mp_img, ts_ms)

                try: 
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                
                self.result_queue.put((raw_image, result, time.time()))

    def get_latest_result(self):
        try:
            return self.result_queue.get(timeout=0.1)
            #return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False