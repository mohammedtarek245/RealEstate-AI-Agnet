"""
Knowledge retrieval module for the Arabic Real Estate AI Agent.
Retrieves relevant knowledge and property matches based on user input.
"""

import logging
import re
from typing import Dict, List

from config import ConversationPhase
from knowledge_base import KnowledgeBase  # ✅ Flat path

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeRetrieval:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base

    def retrieve(self, query: str, phase: ConversationPhase, context: Dict = {}) -> Dict:
        logger.info(f"🔍 Retrieving knowledge for phase: {phase.name}")

        phase_knowledge = self._get_phase_knowledge(phase.name.lower())
        property_results = self._get_matching_properties(query, context)

        result = {"phase_knowledge": phase_knowledge}
        if property_results:
            result["relevant_properties"] = property_results
        return result

    def _get_phase_knowledge(self, phase_name: str) -> Dict:
        return self.knowledge_base.get_phase_knowledge(phase_name)

    def _get_matching_properties(self, query: str, context: Dict) -> List[Dict]:
        filters = {}

        user_info = context.get("user_info", {})

        # 🧠 Add user-provided filters
        mapping = {
            "location": "location",
            "property_type": "type",
            "budget": "price",
            "features": "features",
            "bedrooms": "bedrooms"
        }
        for user_key, prop_key in mapping.items():
            val = user_info.get(user_key)
            if val:
                filters[prop_key] = val

        # 🧠 Try extracting more filters from the query directly
        filters.update(self._extract_filters_from_query(query))

        return self.knowledge_base.get_properties(filters, limit=3)

    def _extract_filters_from_query(self, query: str) -> Dict:
        filters = {}

        # Location extraction (basic Arabic regex)
        loc_match = re.search(r'في\s+([\u0600-\u06FF\s]{2,})', query)
        if loc_match:
            filters["location"] = loc_match.group(1).strip()

        # Property type
        for t in ["شقة", "فيلا", "دوبلكس", "استوديو", "محل", "مكتب"]:
            if t in query:
                filters["type"] = t
                break

        # Budget
        price_match = re.search(r'(\d[\d,]*)\s*-\s*(\d[\d,]*)', query)
        if price_match:
            filters["price"] = f"{price_match.group(1)}-{price_match.group(2)}"

        # Features
        features = []
        for f in ["مسبح", "حديقة", "شرفة", "تكييف", "مفروش", "أمن"]:
            if f in query:
                features.append(f)
        if features:
            filters["features"] = ",".join(features)

        return filters
