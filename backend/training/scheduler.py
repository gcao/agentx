import asyncio
from datetime import timedelta

class TrainingScheduler:
    def __init__(self, model, interval_hours=24):
        self.model = model
        self.interval = timedelta(hours=interval_hours)
        self._task = None

    def start(self):
        """Start the periodic training task"""
        self._task = asyncio.create_task(self._run_periodic())

    async def _run_periodic(self):
        """Run training periodically with the given interval"""
        while True:
            await asyncio.sleep(self.interval.total_seconds())
            await self.model.train()  # Assuming the model has a train method

    def stop(self):
        """Stop the periodic training task"""
        if self._task:
            self._task.cancel()
