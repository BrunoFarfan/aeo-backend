from typing import List
from decouple import config
from langchain_openai import OpenAIEmbeddings


EMBEDDING_MODEL = "text-embedding-3-small"

class EmbeddingService:
    """Service to handle OpenAI embeddings using LangChain."""

    def __init__(self):
        self.openai_api_key = config('OPENAI_API_KEY', default=None)
        self.embeddings = None
        
        if self.openai_api_key:
            try:
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=self.openai_api_key,
                    model=EMBEDDING_MODEL
                )
            except Exception as e:
                print(f'Error initializing OpenAI embeddings: {str(e)}')

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a given text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector

        """
        try:
            if not self.embeddings:
                print('OpenAI API key not configured. Cannot generate embeddings.')
                return []
            
            # Get embedding using LangChain
            embedding = await self.embeddings.aembed_query(text)
            return embedding
            
        except Exception as e:
            print(f'Error generating embedding: {str(e)}')
            return [] 