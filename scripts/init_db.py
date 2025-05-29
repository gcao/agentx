import chromadb
import os

def initialize_database():
    """Initializes the ChromaDB database for long-term memory."""
    print("Initializing ChromaDB for long-term memory...")
    
    # Ensure the models directory exists for ChromaDB storage
    db_path = os.path.join("models", "chroma_db")
    os.makedirs(db_path, exist_ok=True)

    client = chromadb.PersistentClient(path=db_path)
    
    try:
        # Get or create the collection for long-term memory
        collection = client.get_or_create_collection(name="long_term_memory")
        print(f"Successfully initialized ChromaDB collection: {collection.name}")
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        sys.exit(1)

if __name__ == "__main__":
    initialize_database()
