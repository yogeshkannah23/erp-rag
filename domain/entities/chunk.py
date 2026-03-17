"""
Domain entity representing a text chunk with metadata.
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Chunk:
    text: str
    metadata: Dict = field(default_factory=dict)
