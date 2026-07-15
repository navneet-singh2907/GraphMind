from langchain_core.embeddings import Embeddings
from openai import OpenAI
from src.utils.config import NEBIUS_API_KEY, NEBIUS_BASE_URL, EMBEDDING_MODEL


class NebiusEmbeddings(Embeddings):
    def __init__(self):
        self.client = OpenAI(api_key=NEBIUS_API_KEY, base_url=NEBIUS_BASE_URL)
        self.model = EMBEDDING_MODEL

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
