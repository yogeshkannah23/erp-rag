"""
Domain entities for RAG query results.
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Source:
    content: str
    project: str
    domain: str


@dataclass
class QueryResult:
    answer: str
    sources: List[Source] = field(default_factory=list)
    extracted_tags: Dict = field(default_factory=dict)
