from backend.memory.long_term import LongTermMemory
from backend.memory.short_term import ShortTermMemory

class MemoryManager:
    def __init__(self):
        self.long_term_memory = LongTermMemory()
        self.short_term_memory = ShortTermMemory()
        print("MemoryManager initialized with LongTermMemory and ShortTermMemory.")

    async def store_message(self, session_id, message, is_long_term=False):
        """Stores a message in short-term memory and optionally in long-term memory."""
        await self.short_term_memory.store_context(session_id, message)
        if is_long_term:
            await self.long_term_memory.store_conversation([message]) # Assuming store_conversation takes a list of messages

    async def get_recent_messages(self, session_id, limit=10):
        """Retrieves recent messages from short-term memory."""
        return await self.short_term_memory.get_recent_context(session_id, limit)

    async def retrieve_from_long_term(self, query, k=5):
        """Retrieves similar information from long-term memory."""
        return await self.long_term_memory.retrieve_similar(query, k)

    # Add other methods as needed for memory management, e.g., summarization, episodic memory

    async def store_observation(self, observation_text, observation_image=None):
        """Stores an observation from camera or other sensors"""
        session_id = "camera_observations"  # Special session for camera observations
        await self.short_term_memory.store_context(session_id, observation_text)
        
        if observation_image:
            # Store image in long-term memory if needed
            # Implementation depends on your long-term memory storage
            pass
