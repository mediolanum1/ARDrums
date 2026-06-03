import cv2
import threading
import queue

class CameraManager:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._camera_thread, daemon=True).start()

    def _camera_thread(self):
        while self.running:
            success, image = self.cap.read()
            if not success: 
                continue
            image = cv2.flip(image, 1)
            
            try: 
                self.frame_queue.get_nowait()
            except queue.Empty: 
                pass
            
            self.frame_queue.put(image)

    def get_latest_frame(self):
        try: 
            return self.frame_queue.get(timeout=0.1)
           # return self.frame_queue.get_nowait()
        except queue.Empty: 
            return None
        
    def stop(self):
        self.running = False
        self.cap.release()