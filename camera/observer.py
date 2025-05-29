import cv2
import threading
import time
from PIL import Image
import io

class CameraObserver:
    def __init__(self, camera_config, model, memory_manager, capture_interval=30):
        self.camera_config = camera_config
        self.model = model
        self.tokenizer = model.tokenizer  # Store tokenizer from model
        self.memory_manager = memory_manager
        self.capture_interval = capture_interval
        self.latest_frame = None
        self.running = False
        
    def start(self):
        self.running = True
        thread = threading.Thread(target=self._capture_loop)
        thread.daemon = True
        thread.start()
    
    def _capture_loop(self):
        # Handle both device number (int) or URL (str)
        if isinstance(self.camera_config, int):
            cap = cv2.VideoCapture(self.camera_config)
        else:
            cap = cv2.VideoCapture(str(self.camera_config))
            
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera: {self.camera_config}")
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.latest_frame = frame
                # Process and analyze the frame
                # Run async analysis in a new thread to maintain sync context
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._analyze_frame(frame))
                loop.close()
            
            time.sleep(self.capture_interval)
        
        cap.release()
    
    async def _analyze_frame(self, frame):
        # Convert to PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Generate observation using chat interface
        prompt = "Describe what you see in this image. Note any changes or interesting elements."
        observation, _ = self.model.chat(
            self.tokenizer,
            query=prompt,
            history=None,
            images=[pil_image]
        )
        
        # Store observation in memory
        await self.memory_manager.store_observation(observation, pil_image)
    
    def get_current_view(self):
        if self.latest_frame is not None:
            rgb_frame = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_frame)
        return None
