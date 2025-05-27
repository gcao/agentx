import asyncio
from backend.models.model_loader import MultiModalModel
from backend.memory.memory_manager import MemoryManager
from camera.observer import CameraObserver
from backend.training.scheduler import TrainingScheduler

async def main():
    # Initialize components
    model = MultiModalModel()
    memory_manager = MemoryManager()
    camera_observer = CameraObserver("http://your-ip-camera-url/stream")
    training_scheduler = TrainingScheduler(model, interval_hours=24)
    
    # Start services
    camera_observer.start()
    training_scheduler.start()
    
    # Start API server
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    asyncio.run(main())