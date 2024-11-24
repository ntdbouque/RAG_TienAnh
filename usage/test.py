import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.embedding import RAG
from src.settings import Settings

setting = Settings()

rag = RAG(setting)

q = "What is Gesture  language  or  Sign  language ?"

print(rag.contextual_rag_search(q))