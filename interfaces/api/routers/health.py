"""
Routers: /health and /milvus-probe
"""
import logging
import os
import socket
from typing import Optional

from fastapi import APIRouter, Depends

from infrastructure.vector_store.milvus_vector_store import MilvusVectorStore
from interfaces.api.dependencies import get_vector_store

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
def health(vector_store: MilvusVectorStore = Depends(get_vector_store)):
    milvus_ok, milvus_msg = vector_store.health_check()
    return {
        "status": "ok" if milvus_ok else "degraded",
        "milvus": {"connected": milvus_ok, "detail": milvus_msg},
    }


@router.get("/milvus-probe")
def milvus_probe():
    """
    Try every likely Milvus hostname/IP and return which ones are reachable.
    Useful for debugging connection issues in Docker / ERP setups.
    """
    from pymilvus import connections as pym_connections, utility

    port = int(os.getenv("MILVUS_PORT", "19530"))

    candidates = [
        os.getenv("MILVUS_HOST", "localhost"),
        "localhost",
        "127.0.0.1",
        "milvus",
        "milvus-standalone",
        "54.221.211.87",
    ]
    seen: set = set()
    unique_candidates = []
    for h in candidates:
        h = h.strip()
        if h and h not in seen:
            seen.add(h)
            unique_candidates.append(h)

    results = []
    first_working: Optional[dict] = None

    for host in unique_candidates:
        alias = f"probe_{host.replace('.', '_').replace('-', '_')}"
        entry: dict = {
            "host": host,
            "port": port,
            "connected": False,
            "error": None,
            "collections": None,
        }

        try:
            sock = socket.create_connection((host, port), timeout=3)
            sock.close()
        except Exception as tcp_err:
            entry["error"] = f"TCP unreachable — {tcp_err}"
            results.append(entry)
            continue

        try:
            pym_connections.connect(alias=alias, host=host, port=port, timeout=5)
            cols = utility.list_collections(using=alias)
            pym_connections.disconnect(alias)
            entry["connected"] = True
            entry["collections"] = cols
            if first_working is None:
                first_working = {"host": host, "port": port}
        except Exception as e:
            entry["error"] = str(e)
            try:
                pym_connections.disconnect(alias)
            except Exception:
                pass

        results.append(entry)

    return {
        "first_working": first_working,
        "probe_results": results,
        "hint": (
            f"Set MILVUS_HOST={first_working['host']} in your .env"
            if first_working
            else "No working connection found — is Milvus running and on the same Docker network?"
        ),
    }
