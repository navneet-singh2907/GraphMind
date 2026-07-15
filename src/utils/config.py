from dotenv import load_dotenv
import os

load_dotenv()

NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
