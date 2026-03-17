"""
Port (interface) for domain classification.
"""
from abc import ABC, abstractmethod
from typing import List, Optional


class DomainClassifierPort(ABC):
    @abstractmethod
    def classify(self, query: str, available_domains: Optional[List[str]] = None) -> dict:
        """
        Classify a query into one or more domains.
        Returns dict with keys: 'domains' (List[str]) and 'technologies' (List[str]).
        """
