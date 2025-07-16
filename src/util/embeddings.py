from typing import List
from sentence_transformers import SentenceTransformer


class LocalEmbeddings:
    """Local embedding service using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the local embedding model.
        
        Args:
            model_name: Name of the sentence-transformer model to use.
                       Default is 'all-MiniLM-L6-v2' which produces 384-dimensional embeddings.
        """
        self.model = SentenceTransformer(model_name)
        self.vector_dimensions = self.model.get_sentence_embedding_dimension()
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()