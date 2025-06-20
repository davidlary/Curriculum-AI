import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Qdrant
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    import qdrant_client
    from qdrant_client.models import Distance, VectorParams
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.info("Install with: pip install langchain openai qdrant-client")
    exit(1)

CHUNKS_DIR = Path("Chunks")
EMBEDDINGS_DIR = Path("Embeddings")
EMBEDDINGS_DIR.mkdir(exist_ok=True)

class ChunkEmbedder:
    def __init__(self, openai_api_key: str = None, provider: str = "openai"):
        """Initialize the chunk embedder with embeddings and Qdrant vector store."""
        # Check environment variables for API keys
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.xai_api_key = os.getenv("XAI_API_KEY")
        
        self.provider = provider.lower()
        
        # Initialize embeddings based on provider
        if self.provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
            self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        else:
            # Default to OpenAI for embeddings (most providers don't have embedding models)
            if not self.openai_api_key:
                logger.warning("OpenAI API key required for embeddings even with other providers")
                raise ValueError("OpenAI API key required for embeddings. Set OPENAI_API_KEY environment variable.")
            self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Initialize Qdrant client
        self.qdrant_client = qdrant_client.QdrantClient(":memory:")
        
    def load_chunks(self, discipline: str = "Physics") -> List[Dict[str, Any]]:
        """Load chunks from JSONL files for a specific discipline."""
        chunks = []
        chunk_files = list(CHUNKS_DIR.glob(f"{discipline}_*.jsonl"))
        
        if not chunk_files:
            logger.warning(f"No chunk files found for discipline: {discipline}")
            return chunks
            
        for chunk_file in chunk_files:
            logger.info(f"Loading chunks from: {chunk_file}")
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            chunk = json.loads(line.strip())
                            chunks.append(chunk)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON at line {line_num} in {chunk_file}: {e}")
            except Exception as e:
                logger.error(f"Error reading {chunk_file}: {e}")
                
        logger.info(f"Loaded {len(chunks)} chunks for discipline: {discipline}")
        return chunks
    
    def create_documents(self, chunks: List[Dict[str, Any]]) -> List[Document]:
        """Convert chunks to LangChain Document objects."""
        documents = []
        
        for chunk in chunks:
            if not chunk.get('text', '').strip():
                continue
                
            # Create metadata
            metadata = {
                'discipline': chunk.get('discipline', ''),
                'language': chunk.get('language', ''),
                'level': chunk.get('level', ''),
                'source_file': chunk.get('source_file', ''),
                'chunk_index': chunk.get('chunk_index', 0)
            }
            
            # Create document
            doc = Document(
                page_content=chunk['text'],
                metadata=metadata
            )
            documents.append(doc)
            
        logger.info(f"Created {len(documents)} documents")
        return documents
    
    def create_vector_store(self, documents: List[Document], collection_name: str) -> Qdrant:
        """Create Qdrant vector store from documents."""
        logger.info(f"Creating vector store: {collection_name}")
        
        # Create collection
        self.qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        
        # Create vector store
        vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name=collection_name,
            embeddings=self.embeddings
        )
        
        # Add documents to vector store
        logger.info("Adding documents to vector store...")
        vectorstore.add_documents(documents)
        
        logger.info(f"Vector store created with {len(documents)} documents")
        return vectorstore
    
    def save_embeddings_metadata(self, discipline: str, num_chunks: int, collection_name: str):
        """Save metadata about the embeddings."""
        metadata = {
            'discipline': discipline,
            'num_chunks': num_chunks,
            'collection_name': collection_name,
            'embedding_model': 'text-embedding-ada-002',
            'vector_store': 'qdrant',
            'created_at': str(Path().cwd())
        }
        
        metadata_file = EMBEDDINGS_DIR / f"{discipline}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved embeddings metadata to: {metadata_file}")

def embed_chunks(discipline: str = "Physics", openai_api_key: str = None, provider: str = "openai"):
    """Main function to embed chunks for a discipline."""
    logger.info(f"Starting embedding process for discipline: {discipline}")
    
    try:
        # Initialize embedder (will automatically use environment variables if available)
        embedder = ChunkEmbedder(openai_api_key=openai_api_key, provider=provider)
        
        # Load chunks
        chunks = embedder.load_chunks(discipline)
        if not chunks:
            logger.error(f"No chunks found for discipline: {discipline}")
            return
        
        # Create documents
        documents = embedder.create_documents(chunks)
        if not documents:
            logger.error("No valid documents created from chunks")
            return
        
        # Create vector store
        collection_name = f"{discipline.lower()}_curriculum"
        vectorstore = embedder.create_vector_store(documents, collection_name)
        
        # Save metadata
        embedder.save_embeddings_metadata(discipline, len(documents), collection_name)
        
        # Test similarity search
        logger.info("Testing similarity search...")
        test_query = "What is physics?"
        results = vectorstore.similarity_search(test_query, k=3)
        logger.info(f"Found {len(results)} similar documents for test query")
        
        logger.info(f"✅ Successfully embedded {len(documents)} chunks for {discipline}")
        
    except Exception as e:
        logger.error(f"Error during embedding process: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embed textbook chunks using AI embeddings')
    parser.add_argument('--discipline', '-d', default='Physics', 
                       help='Discipline to embed (default: Physics)')
    parser.add_argument('--openai-api-key', 
                       help='OpenAI API key (optional - uses OPENAI_API_KEY env var by default)')
    parser.add_argument('--provider', default='openai', choices=['openai'],
                       help='AI provider for embeddings (default: openai)')
    
    args = parser.parse_args()
    
    embed_chunks(discipline=args.discipline, openai_api_key=args.openai_api_key, provider=args.provider)