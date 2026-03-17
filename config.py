"""
Configuration for PDF Vector Service — reads from environment variables.
"""
import re
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = "gpt-4o-mini"
LLM_TEMPERATURE: float = 0.7

# ── Embeddings ─────────────────────────────────────────────────────────────────
HUGGINGFACE_API_TOKEN: str = os.getenv("HUGGINGFACE_API_TOKEN", "")
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
ENABLE_SUMMARIZATION: bool = True


# ── Search ─────────────────────────────────────────────────────────────────────
DEFAULT_SEARCH_K: int = 4
CHUNKS_PER_PROJECT: int = 3
MAX_TOTAL_CHUNKS: int = 20
DEFAULT_DOMAINS = ["Migration", "Security", "AI", "Frontend", "Backend", "DevOps"]

# ── Milvus ─────────────────────────────────────────────────────────────────────
def get_milvus_connection_args() -> dict:
    host = os.getenv("MILVUS_HOST", "localhost").strip()
    port = os.getenv("MILVUS_PORT", "19530").strip()
    return {"uri": f"http://{host}:{port}"}

def check_milvus_connection() -> tuple[bool, str]:
    """Returns (ok, message). Tries a live gRPC connection to Milvus."""
    from pymilvus import connections, utility
    host = os.getenv("MILVUS_HOST", "localhost").strip()
    port = os.getenv("MILVUS_PORT", "19530").strip()
    alias = "health_check"
    try:
        connections.connect(alias=alias, host=host, port=port, timeout=5)
        utility.list_collections(using=alias)
        connections.disconnect(alias)
        return True, f"Connected to Milvus at {host}:{port}"
    except Exception as e:
        return False, f"Milvus unreachable at {host}:{port} — {e}"

def get_collection_name(domain: str) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', domain.lower())
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return f"{sanitized}_case_studies"

# ── Prompts ────────────────────────────────────────────────────────────────────
SUMMARIZATION_PROMPT = """
You are summarizing a technical project document. Combine these 4 sections into a single cohesive summary.

Requirements:
- Preserve ALL information, keep it concise (aim for 60% of original length)
- Flow: Problem → Features → Tech Stack → Challenges/Solutions
- Keep ALL technical terms and specific details

1. Business Problem:
{business_problem}

2. Features/Modules Delivered:
{features}

3. Tech Stack Used:
{tech_stack}

4. Key Challenges Solved:
{challenges}

Provide a comprehensive summary combining all sections.
"""



RAG_PROMPT_TEMPLATE = """
You are a helpful technical assistant. Answer the question based on the provided context.
The context contains information from multiple projects.

Guidelines:
1. Start with: "We have X projects related to [topic]: Project A, Project B, ..."
2. For each project: **Project Name**: brief description with technologies used.
3. End with a brief comparison or conclusion if applicable.
4. Use **bold** for project names. Do NOT use markdown headers (###).
5. Only include projects relevant to the question.

Context:
{context}

Question:
{question}

Answer:
"""
