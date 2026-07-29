from langchain_core.embeddings import Embeddings
from openai import OpenAI
from src.utils.config import (
    EMBEDDING_MODEL,
    MODEL_MAX_RETRIES,
    MODEL_TIMEOUT_SECONDS,
    NEBIUS_API_KEY,
    NEBIUS_BASE_URL,
)


class NebiusEmbeddings(Embeddings):
    def __init__(self):
        self.client = OpenAI(
            api_key=NEBIUS_API_KEY,
            base_url=NEBIUS_BASE_URL,
            timeout=MODEL_TIMEOUT_SECONDS,
            max_retries=MODEL_MAX_RETRIES,
        )
        self.model = EMBEDDING_MODEL

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
