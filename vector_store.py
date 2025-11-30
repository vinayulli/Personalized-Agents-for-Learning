import os
import httpx
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Custom Embedding Class for OpenRouter BGE-M3
class OpenRouterEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "baai/bge-m3"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/embeddings"
        self.client = httpx.Client(headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:8501", # Optional, for OpenRouter stats
            "X-Title": "Personalized Learning Agent" # Optional
        }, timeout=60.0)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        print(f"DEBUG: Embedding batch of {len(texts)} texts...")
        embeddings = []
        for text in texts:
            try:
                response = self.client.post(
                    self.base_url,
                    json={
                        "model": self.model,
                        "input": text
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and len(data['data']) > 0:
                        embeddings.append(data['data'][0]['embedding'])
                    else:
                         raise Exception(f"Empty embedding data: {data}")
                else:
                    print(f"ERROR: OpenRouter API Error: {response.text}")
                    raise Exception(f"OpenRouter API Error: {response.text}")
            except Exception as e:
                print(f"ERROR: Embedding chunk failed: {e}")
                raise e
        print(f"DEBUG: Successfully embedded {len(embeddings)} texts.")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        print(f"DEBUG: Embedding query: '{text[:50]}...'")
        return self.embed_documents([text])[0]

# Initialize Embedding Model
def get_embeddings():
    return OpenRouterEmbeddings(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="baai/bge-m3"
    )

# Initialize Qdrant Client
def get_qdrant_client():
    url = os.getenv("QDRANT_URL")
    print(f"DEBUG: Connecting to Qdrant at {url}...")
    return QdrantClient(
        url=url,
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=False 
    )

# Get Vector Store
def get_vector_store(collection_name="knowledge_base"):
    embeddings = get_embeddings()
    client = get_qdrant_client()
    
    print(f"DEBUG: Checking if collection '{collection_name}' exists...")
    if not client.collection_exists(collection_name):
        print(f"DEBUG: Collection '{collection_name}' does not exist. Creating it now...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        print(f"DEBUG: Collection '{collection_name}' created successfully.")
    else:
        print(f"DEBUG: Collection '{collection_name}' found.")

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

def ingest_text(text: str, metadata: dict = None, collection_name="knowledge_base", status_callback=None):
    print("DEBUG: Starting text ingestion...")
    
    # 1. Split text
    text_splitter = get_text_splitter()
    texts = text_splitter.split_text(text)
    
    if not texts:
        print("DEBUG: No text to ingest.")
        return 0
    
    print(f"DEBUG: Split text into {len(texts)} chunks.")

    # 2. Create metadatas list
    metadatas = [metadata] * len(texts) if metadata else None
    
    # 3. Get Vector Store (handles collection creation)
    vector_store = get_vector_store(collection_name)
    
    batch_size = 5
    total_chunks = len(texts)
    
    print(f"DEBUG: Starting batch ingestion of {total_chunks} chunks...")
    
    for i in range(0, total_chunks, batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_metadatas = metadatas[i : i + batch_size] if metadatas else None
        
        print(f"DEBUG: Ingesting batch {i}-{i+batch_size}...")
        vector_store.add_texts(texts=batch_texts, metadatas=batch_metadatas)
        
        if status_callback:
            progress = min(i + batch_size, total_chunks) / total_chunks
            status_callback(progress, f"Ingesting chunk {min(i + batch_size, total_chunks)}/{total_chunks}...")
            
    print(f"DEBUG: Successfully ingested {len(texts)} chunks into {collection_name}")
    return len(texts)
