import cv2
import threading
import time
from PIL import Image
import io

class CameraObserver:
    def __init__(self, camera_url, capture_interval=30):
        self.camera_url = camera_url
        self.capture_interval = capture_interval
        self.latest_frame = None
        self.running = False
        
    def start(self):
        self.running = True
        thread = threading.Thread(target=self._capture_loop)
        thread.daemon = True
        thread.start()
    
    def _capture_loop(self):
        cap = cv2.VideoCapture(self.camera_url)
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.latest_frame = frame
                # Process and analyze the frame
                self._analyze_frame(frame)
            
            time.sleep(self.capture_interval)
        
        cap.release()
    
    def _analyze_frame(self, frame):
        # Convert to PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Generate observation
        observation = model.generate(
            "Describe what you see in this image. Note any changes or interesting elements.",
            pil_image
        )
        
        # Store observation in memory
        memory_manager.store_observation(observation, pil_image)
    
    def get_current_view(self):
        if self.latest_frame is not None:
            rgb_frame = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_frame)
        return None