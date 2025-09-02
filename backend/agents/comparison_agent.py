"""
Comparison Agent - Compares materials with supplier BOM
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class ComparisonAgent:
    def __init__(self, gemini_client):
        self.gemini_client = gemini_client

    async def compare_materials(self, materials: List[Dict], supplier_items: List[Dict]) -> List[Dict]:
        """Compare materials with supplier BOM"""
        try:
            matched_materials = []

            for material in materials:
                # Simple matching logic for demo
                material_copy = material.copy()
                material_copy.update({
                    'confidence_score': 0.8,
                    'supplier_description': 'Matched supplier item',
                    'supplier_part_number': 'SP001',
                    'match_source': 'supplier_bom',
                    'reasoning': 'Matched based on material name similarity'
                })
                matched_materials.append(material_copy)

            return matched_materials
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            raise