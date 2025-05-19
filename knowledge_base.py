"""
Knowledge Base module for Arabic Real Estate AI Agent.
Loads structured CSVs and JSONs into memory from Colab, Hugging Face Spaces, or local disk.
"""

import os
import json
import csv
import logging
from typing import List, Dict, Optional

import pandas as pd  # Needed for injected DataFrame support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self, csv_data: Optional[Dict[str, pd.DataFrame]] = None, rules_data: Optional[Dict] = None):
        logger.info("🔍 Initializing Knowledge Base...")

        self.injected_mode = bool(csv_data or rules_data)
        self.data = {}

        if self.injected_mode:
            # Load from passed-in Colab/Spaces data
            self.properties = csv_data.get("properties", pd.DataFrame()).to_dict(orient="records")
            self.rules = rules_data or {}
            self.phase_knowledge = self._build_phase_knowledge_from_json(self.rules)
            logger.info("✅ Loaded knowledge from injected data.")
        else:
            # Load from files
            self.properties = self._load_csv("properties.csv")
            self.rules = self._load_json("rules.json")
            self.phase_knowledge = self._load_phase_knowledge()
            logger.info("✅ Loaded knowledge from local files.")

        logger.info(f"🏘️ Loaded {len(self.properties)} properties.")
        logger.info(f"📏 Loaded {len(self.rules.get('budget_advice', []))} budget rules.")

    def _load_csv(self, filename: str) -> List[Dict]:
        if not os.path.exists(filename):
            logger.warning(f"⚠️ CSV file not found: {filename}")
            return []
        with open(filename, encoding='utf-8') as f:
            return list(csv.DictReader(f))

    def _load_json(self, filename: str) -> Dict:
        if not os.path.exists(filename):
            logger.warning(f"⚠️ JSON file not found: {filename}")
            return {}
        with open(filename, encoding='utf-8') as f:
            return json.load(f)

    def _load_phase_knowledge(self) -> Dict[str, Dict]:
        phases = ["discovery", "summary", "suggestion", "persuasion", "alternative", "urgency", "closing"]
        knowledge = {}
        for phase in phases:
            fname = f"{phase}.json"
            if os.path.exists(fname):
                knowledge[phase] = self._load_json(fname)
            else:
                logger.warning(f"⚠️ Missing phase file: {fname}")
        return knowledge

    def _build_phase_knowledge_from_json(self, rules: Dict) -> Dict[str, Dict]:
        # Simulate per-phase knowledge from rules.json if individual files are not provided
        return {
            "discovery": {
                "suggested_questions": [
                    "ايه نوع العقار اللي بتدور عليه؟", "فين تحب يكون المكان؟", "الميزانية تقريبا كام؟"
                ]
            },
            "summary": {
                "confirmation_phrases": ["تمام", "مظبوط", "موافق"]
            },
            "suggestion": {
                "call_to_action": "شوف العقارات دي وقلّي رأيك"
            }
            # The rest can be expanded as needed
        }

    def get_phase_knowledge(self, phase_name: str) -> Dict:
        return self.phase_knowledge.get(phase_name.lower(), {})

    def get_properties(self, filters: Dict = {}, limit: int = 5) -> List[Dict]:
        matches = []
        for prop in self.properties:
            match = True
            for key, value in filters.items():
                if key not in prop:
                    continue
                if isinstance(value, str):
                    if value.lower() not in prop[key].lower():
                        match = False
                        break
                elif isinstance(value, (list, tuple)):
                    if not any(v.lower() in prop[key].lower() for v in value):
                        match = False
                        break
            if match:
                matches.append(prop)
            if len(matches) >= limit:
                break
        return matches

    def get_rules(self) -> Dict:
        return self.rules
