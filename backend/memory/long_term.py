from chromadb import Client
from sentence_transformers import SentenceTransformer
import uuid

class LongTermMemory:
    def __init__(self):
        self.client = Client()
        self.collection = self.client.create_collection("conversations")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def store_conversation(self, messages, metadata=None):
        # Convert conversation to text
        text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        # Generate embedding
        embedding = self.encoder.encode(text).tolist()
        
        # Store in ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}],
            ids=[str(uuid.uuid4())]
        )
    
    async def retrieve_similar(self, query, k=5):
        query_embedding = self.encoder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return results['documents'][0]