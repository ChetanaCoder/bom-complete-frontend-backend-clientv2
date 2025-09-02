import logging
from typing import List, Dict, Tuple
from .knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

class ItemMatcher:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

    def match_items_with_knowledge_base(self, extracted_items: List[Dict], 
                                      supplier_bom: List[Dict], 
                                      workflow_id: str) -> List[Dict]:
        """
        Enhanced matching that leverages knowledge base for better accuracy
        """
        enhanced_matches = []

        # First, add new items to knowledge base
        try:
            new_item_ids = self.kb.add_items(extracted_items, workflow_id)
            logger.info(f"Added {len(new_item_ids)} items to knowledge base for workflow {workflow_id}")
        except Exception as e:
            logger.error(f"Failed to add items to knowledge base: {e}")
            new_item_ids = []

        for i, item in enumerate(extracted_items):
            # FIXED: Use consistent field names - try qa_material_name first, then name
            material_name = item.get('qa_material_name', item.get('name', ''))
            part_number = item.get('part_number', '')

            # Check knowledge base for previous matches
            try:
                kb_matches = self.kb.find_similar_items(material_name, part_number)
            except Exception as e:
                logger.error(f"Failed to find similar items: {e}")
                kb_matches = []

            # Find matches in current supplier BOM
            supplier_matches = self._find_supplier_matches(item, supplier_bom)

            # Combine and rank matches
            best_match = self._select_best_match(item, kb_matches, supplier_matches)

            # Generate reasoning for the match
            reasoning = self._generate_match_reasoning(kb_matches, supplier_matches, best_match)

            # FIXED: Create enhanced match result with consistent field mapping
            match_result = {
                # Preserve all original fields from extracted item
                **item,
                
                # Ensure qa_material_name is always set
                'qa_material_name': material_name,
                
                # Enhanced matching fields
                'knowledge_base_matches': len(kb_matches),
                'supplier_matches': len(supplier_matches),
                'has_previous_match': len(kb_matches) > 0,
                'match_source': self._determine_match_source(kb_matches, supplier_matches),
                'confidence_score': best_match.get('confidence_score', 0.0) if best_match else 0.0,
                'supplier_description': best_match.get('supplier_description', '') if best_match else '',
                'supplier_part_number': best_match.get('supplier_part_number', '') if best_match else '',
                'reasoning': reasoning,
                
                # Additional metadata
                'processing_timestamp': self._get_timestamp(),
                'matcher_version': '2.0',
                'workflow_id': workflow_id
            }

            enhanced_matches.append(match_result)

        logger.info(f"Enhanced matching completed for workflow {workflow_id}: {len(enhanced_matches)} matches")
        return enhanced_matches

    def _find_supplier_matches(self, item: Dict, supplier_bom: List[Dict]) -> List[Dict]:
        """Find matches in current supplier BOM"""
        # FIXED: Use consistent field names
        material_name = item.get('qa_material_name', item.get('name', '')).lower()
        part_number = item.get('part_number', '')

        matches = []

        for supplier_item in supplier_bom:
            supplier_desc = supplier_item.get('description', '').lower()
            supplier_part = supplier_item.get('part_number', '')

            confidence = 0.0

            # Exact part number match
            if part_number and supplier_part and part_number.strip() == supplier_part.strip():
                confidence = 0.95
            # Partial part number match
            elif part_number and supplier_part and (
                part_number.strip() in supplier_part.strip() or 
                supplier_part.strip() in part_number.strip()
            ):
                confidence = 0.8
            # Name similarity (improved word matching)
            elif material_name and supplier_desc:
                common_words = set(material_name.split()) & set(supplier_desc.split())
                if common_words:
                    # Calculate similarity ratio
                    material_words = set(material_name.split())
                    supplier_words = set(supplier_desc.split())
                    jaccard_similarity = len(common_words) / len(material_words | supplier_words)
                    confidence = min(0.85, jaccard_similarity * 0.9)

            if confidence > 0.3:  # Minimum threshold
                matches.append({
                    'supplier_description': supplier_item.get('description', ''),
                    'supplier_part_number': supplier_item.get('part_number', ''),
                    'confidence_score': confidence,
                    'match_type': 'part_number' if confidence > 0.85 else 'description',
                    'supplier_data': supplier_item  # Include full supplier item data
                })

        return sorted(matches, key=lambda x: x['confidence_score'], reverse=True)

    def _select_best_match(self, item: Dict, kb_matches: List[Dict], 
                          supplier_matches: List[Dict]) -> Dict:
        """Select the best match from knowledge base and supplier matches"""

        # Prioritize knowledge base exact matches with high confidence
        for kb_match in kb_matches:
            if kb_match.get('match_type') == 'exact' and kb_match.get('confidence_score', 0) > 0.8:
                return {
                    'id': kb_match.get('id'),
                    'supplier_description': kb_match.get('material_name', ''),
                    'supplier_part_number': kb_match.get('part_number', ''),
                    'confidence_score': min(0.95, kb_match.get('confidence_score', 0.9)),
                    'match_type': 'exact',
                    'match_source': 'knowledge_base'
                }

        # Then prioritize high-confidence supplier matches
        if supplier_matches:
            best_supplier = supplier_matches[0]  # Already sorted by confidence
            if best_supplier.get('confidence_score', 0) > 0.7:
                return {
                    **best_supplier,
                    'match_source': 'supplier_bom'
                }

        # Fall back to knowledge base fuzzy matches
        if kb_matches:
            best_kb = max(kb_matches, key=lambda x: x.get('confidence_score', 0))
            if best_kb.get('confidence_score', 0) > 0.5:
                return {
                    'id': best_kb.get('id'),
                    'supplier_description': best_kb.get('material_name', ''),
                    'supplier_part_number': best_kb.get('part_number', ''),
                    'confidence_score': max(0.6, best_kb.get('confidence_score', 0.6)),
                    'match_type': 'fuzzy',
                    'match_source': 'knowledge_base'
                }

        # Final fallback - try lower confidence supplier matches
        if supplier_matches:
            best_supplier = supplier_matches[0]
            return {
                **best_supplier,
                'match_source': 'supplier_bom_low_confidence'
            }

        return {}

    def _determine_match_source(self, kb_matches: List[Dict], 
                               supplier_matches: List[Dict]) -> str:
        """Determine the primary source of the match"""
        if kb_matches and supplier_matches:
            return 'hybrid'
        elif kb_matches:
            return 'knowledge_base'
        elif supplier_matches:
            return 'supplier_bom'
        else:
            return 'no_match'

    def _generate_match_reasoning(self, kb_matches: List[Dict], 
                                 supplier_matches: List[Dict], 
                                 best_match: Dict) -> str:
        """Generate human-readable reasoning for the match"""
        if not best_match:
            if not kb_matches and not supplier_matches:
                return "No matches found in knowledge base or supplier BOM"
            else:
                return "Matches found but confidence scores below minimum threshold"

        match_source = best_match.get('match_source', 'unknown')
        confidence = best_match.get('confidence_score', 0)
        match_type = best_match.get('match_type', 'unknown')

        if match_source == 'knowledge_base':
            if match_type == 'exact':
                return f"Exact match found in knowledge base from previous workflow (confidence: {confidence:.1%})"
            else:
                return f"Similar item found in knowledge base based on name similarity (confidence: {confidence:.1%})"
        elif match_source == 'supplier_bom':
            if match_type == 'part_number':
                return f"Part number match found in current supplier BOM (confidence: {confidence:.1%})"
            else:
                return f"Description match found in current supplier BOM (confidence: {confidence:.1%})"
        elif match_source == 'supplier_bom_low_confidence':
            return f"Low confidence match found in supplier BOM - requires review (confidence: {confidence:.1%})"
        elif match_source == 'hybrid':
            return f"Match verified through both knowledge base and supplier BOM (confidence: {confidence:.1%})"

        return f"Match found with {confidence:.1%} confidence via {match_source}"

    def _get_timestamp(self) -> str:
        """Get current timestamp for logging"""
        from datetime import datetime
        return datetime.utcnow().isoformat()