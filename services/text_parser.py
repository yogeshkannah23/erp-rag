"""
Text Parser Service — extracts structured sections from PDF text.
"""
import re
from typing import Dict, List, Optional


class TextParser:
    SECTION_PATTERNS = {
        "business_problem": r"(?:1\.\s*)?Business\s+Problem",
        "features": r"(?:2\.\s*)?Features\s*/\s*Modules\s+Delivered",
        "tech_stack": r"(?:3\.\s*)?Tech\s+Stack\s+Used",
        "key_challenges": r"(?:4\.\s*)?Key\s+Challenge(?:s)?\s+Solved",
    }

    def extract_section(self, text: str, section_name: str, next_section_name: Optional[str] = None) -> str:
        if section_name not in self.SECTION_PATTERNS:
            return ""
        pattern = self.SECTION_PATTERNS[section_name]
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if not match:
            return ""
        start_pos = match.end()
        if next_section_name and next_section_name in self.SECTION_PATTERNS:
            next_match = re.search(self.SECTION_PATTERNS[next_section_name], text[start_pos:], re.IGNORECASE | re.MULTILINE)
            if next_match:
                return text[start_pos: start_pos + next_match.start()].strip()
        remaining = text[start_pos:]
        end_match = re.search(r'\n\s*(?:5|6|7|8|9|10)\.\s+[A-Z]', remaining)
        if end_match:
            return remaining[: end_match.start()].strip()
        return remaining[:2000].strip()

    def parse_project_document(self, text: str) -> Dict[str, str]:
        section_order = ["business_problem", "features", "tech_stack", "key_challenges"]
        return {
            name: self.extract_section(text, name, section_order[i + 1] if i + 1 < len(section_order) else None)
            for i, name in enumerate(section_order)
        }

    def identify_sections(self, text: str) -> List[str]:
        return [
            name for name, pattern in self.SECTION_PATTERNS.items()
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        ]

    def has_required_sections(self, text: str) -> bool:
        required = {"business_problem", "features", "tech_stack", "key_challenges"}
        return required.issubset(set(self.identify_sections(text)))
