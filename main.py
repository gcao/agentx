import asyncio
import threading
import yaml
from pathlib import Path
from backend.models.model_loader import MultiModalModel
from backend.memory.memory_manager import MemoryManager
from camera.observer import CameraObserver
from backend.training.scheduler import TrainingScheduler
import uvicorn

# Load config
with open(Path(__file__).parent / "config/agent_config.yaml") as f:
    config = yaml.safe_load(f)

def run_uvicorn():
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000)

async def main():
    try:
        # Initialize components
        model = MultiModalModel()
        memory_manager = MemoryManager()
        camera_observer = CameraObserver(config["camera"]["device"], model, memory_manager)
        training_scheduler = TrainingScheduler(model, interval_hours=24)
        
        # Start services
        camera_observer.start()
        training_scheduler.start()
        
        # Start API server in a separate thread
        api_thread = threading.Thread(target=run_uvicorn, daemon=True)
        api_thread.start()
        
        print("All services started successfully")
        
    except Exception as e:
        print(f"Failed to initialize services: {str(e)}")
        raise
    
    # Keep the main thread alive
    while True:
        await asyncio.sleep(3600)  # Sleep for 1 hour

if __name__ == "__main__":
    asyncio.run(main())
