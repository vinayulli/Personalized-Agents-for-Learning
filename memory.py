import os
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv()

def get_memory_client():
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "user_memories",
                "url": os.getenv("QDRANT_URL"),
                "api_key": os.getenv("QDRANT_API_KEY"),
                "embedding_model_dims": 1024, 
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "baai/bge-m3",
                "openai_base_url": "https://openrouter.ai/api/v1",
                "api_key": os.getenv("OPENROUTER_API_KEY"),
            }
        }
    }
    
    return Memory.from_config(config)

def add_memory(user_id, text):
    print(f"DEBUG: Adding memory for user {user_id}")
    m = get_memory_client()
    m.add(text, user_id=user_id)
    print("DEBUG: Memory added.")

def get_memories(user_id, query=None):
    print(f"DEBUG: Fetching memories for user {user_id} with query '{query}'")
    m = get_memory_client()
    if query:
        results = m.search(query, user_id=user_id)
        print(f"DEBUG: Found {len(results)} memories.")
        return results
    return m.get_all(user_id=user_id)
