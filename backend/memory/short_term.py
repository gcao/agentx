import redis
import json
from datetime import datetime, timedelta

class ShortTermMemory:
    def __init__(self):
        self.redis_client = redis.Redis(decode_responses=True)
        self.ttl = 3600  # 1 hour TTL for short-term memory
    
    async def store_context(self, session_id, message):
        key = f"context:{session_id}"
        self.redis_client.rpush(key, json.dumps(message))
        self.redis_client.expire(key, self.ttl)
    
    async def get_recent_context(self, session_id, limit=10):
        key = f"context:{session_id}"
        messages = self.redis_client.lrange(key, -limit, -1)
        return [json.loads(m) for m in messages]
    
    async def summarize_context(self, messages):
        # Use the model to create a summary
        context = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        summary = model.generate(f"Summarize this conversation: {context}")
        return summary