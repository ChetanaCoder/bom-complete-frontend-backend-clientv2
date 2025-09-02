import sqlite3
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the knowledge base database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Items table for storing previously processed items
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    material_name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    part_number TEXT,
                    vendor_name TEXT,
                    classification_label INTEGER,
                    classification_confidence TEXT,
                    qc_process_step TEXT,
                    unit_of_measure TEXT,
                    quantity TEXT,
                    consumable_jigs_tools BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    workflow_id TEXT,
                    source_document TEXT,
                    qa_excerpt TEXT,
                    category TEXT
                )
            ''')

            # Item matches table for storing successful matches
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS item_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_item_id INTEGER,
                    matched_item_id INTEGER,
                    confidence_score REAL,
                    match_type TEXT,
                    supplier_description TEXT,
                    supplier_part_number TEXT,
                    reasoning TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    workflow_id TEXT,
                    FOREIGN KEY (original_item_id) REFERENCES items (id),
                    FOREIGN KEY (matched_item_id) REFERENCES items (id)
                )
            ''')

            # Create indexes for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_material_name ON items(normalized_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_part_number ON items(part_number)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_workflow_id ON items(workflow_id)')

            conn.commit()
            conn.close()
            logger.info("Knowledge base database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")

    def normalize_material_name(self, material_name: str) -> str:
        """Normalize material name for better matching"""
        if not material_name:
            return ""

        # Convert to lowercase, remove extra spaces, remove special characters
        normalized = material_name.lower().strip()
        # Add more normalization rules as needed
        normalized = ''.join(char for char in normalized if char.isalnum() or char.isspace())
        normalized = ' '.join(normalized.split())  # Remove multiple spaces

        return normalized

    def add_items(self, items: List[Dict], workflow_id: str, source_document: str = None) -> List[int]:
        """Add new items to the knowledge base"""
        if not items:
            return []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            added_ids = []

            for item in items:
                normalized_name = self.normalize_material_name(item.get('qa_material_name', ''))

                cursor.execute('''
                    INSERT INTO items (
                        material_name, normalized_name, part_number, vendor_name,
                        classification_label, classification_confidence, qc_process_step,
                        unit_of_measure, quantity, consumable_jigs_tools, workflow_id, 
                        source_document, qa_excerpt, category
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    item.get('qa_material_name', ''),
                    normalized_name,
                    item.get('part_number'),
                    item.get('vendor_name'),
                    item.get('qa_classification_label'),
                    item.get('qa_confidence_level'),
                    item.get('qc_process_step'),
                    item.get('unit_of_measure'),
                    item.get('quantity'),
                    item.get('consumable_jigs_tools', False),
                    workflow_id,
                    source_document,
                    item.get('qa_excerpt', ''),
                    item.get('category', '')
                ))

                added_ids.append(cursor.lastrowid)

            conn.commit()
            conn.close()

            return added_ids
        except Exception as e:
            logger.error(f"Failed to add items to knowledge base: {e}")
            return []

    def find_similar_items(self, material_name: str, part_number: str = None, threshold: float = 0.7) -> List[Dict]:
        """Find similar items in the knowledge base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            normalized_query = self.normalize_material_name(material_name)

            # Try exact match first
            results = []
            if normalized_query:
                cursor.execute('''
                    SELECT * FROM items 
                    WHERE normalized_name = ? 
                    ORDER BY created_at DESC
                    LIMIT 5
                ''', (normalized_query,))
                exact_matches = cursor.fetchall()

                # Convert to dictionaries
                columns = [
                    'id', 'material_name', 'normalized_name', 'part_number', 'vendor_name',
                    'classification_label', 'classification_confidence', 'qc_process_step',
                    'unit_of_measure', 'quantity', 'consumable_jigs_tools', 'created_at', 'updated_at',
                    'workflow_id', 'source_document', 'qa_excerpt', 'category'
                ]

                for match in exact_matches:
                    item_dict = dict(zip(columns, match))
                    item_dict['match_type'] = 'exact'
                    item_dict['confidence_score'] = 1.0
                    results.append(item_dict)

            conn.close()
            return results

        except Exception as e:
            logger.error(f"Failed to find similar items: {e}")
            return []

    def get_processing_stats(self) -> Dict:
        """Get statistics about items processed"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM items')
            total_items = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(DISTINCT workflow_id) FROM items WHERE workflow_id IS NOT NULL')
            total_workflows = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM item_matches')
            total_matches = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(DISTINCT original_item_id) FROM item_matches')
            unique_matched_items = cursor.fetchone()[0]

            conn.close()

            return {
                'total_items': total_items,
                'total_workflows': total_workflows,
                'total_matches': total_matches,
                'unique_matched_items': unique_matched_items,
                'match_rate': (unique_matched_items / total_items * 100) if total_items > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                'total_items': 0,
                'total_workflows': 0,
                'total_matches': 0,
                'unique_matched_items': 0,
                'match_rate': 0
            }

    def clear_all_data(self):
        """Clear all data from knowledge base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('DELETE FROM item_matches')
            cursor.execute('DELETE FROM items')

            conn.commit()
            conn.close()
            logger.info("Knowledge base cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear knowledge base: {e}")