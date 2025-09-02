"""
Supplier BOM Agent - Processes Excel/CSV supplier BOM files
"""

import logging
import pandas as pd
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)

class SupplierBOMAgent:
    def __init__(self, gemini_client):
        self.gemini_client = gemini_client
        self.stats = {"files_processed": 0, "items_extracted": 0, "errors": 0}
        logger.info("Supplier BOM Agent initialized")

    async def process_supplier_bom(self, file_path: str) -> Dict:
        """Process supplier BOM file (Excel or CSV)"""
        try:
            logger.info(f"Processing supplier BOM: {file_path}")

            # Read the file
            if file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise Exception(f"Unsupported file format: {file_path}")

            # Process the data
            items = self._extract_bom_items(df)

            # Update statistics
            self.stats["files_processed"] += 1
            self.stats["items_extracted"] += len(items)

            return {
                "success": True,
                "items": items,
                "total_items": len(items),
                "columns": list(df.columns),
                "processing_stats": self.stats.copy()
            }

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Supplier BOM processing error: {e}")
            
            # Return demo data on error
            demo_items = self._create_demo_supplier_items()
            return {
                "success": False,
                "error": str(e),
                "items": demo_items,
                "total_items": len(demo_items),
                "processing_stats": self.stats.copy()
            }

    def _extract_bom_items(self, df: pd.DataFrame) -> List[Dict]:
        """Extract items from DataFrame"""
        items = []
        
        # Common column mappings
        column_mappings = {
            'description': ['description', 'item_description', 'product_name', 'name'],
            'part_number': ['part_number', 'part_no', 'item_code', 'sku'],
            'quantity': ['quantity', 'qty', 'amount'],
            'unit_price': ['unit_price', 'price', 'cost'],
            'supplier_name': ['supplier', 'vendor', 'manufacturer'],
            'category': ['category', 'type', 'class']
        }

        # Find actual column names
        actual_columns = {}
        for field, possible_names in column_mappings.items():
            for col_name in df.columns:
                if col_name.lower() in [name.lower() for name in possible_names]:
                    actual_columns[field] = col_name
                    break

        # Extract data
        for _, row in df.iterrows():
            item = {
                'description': str(row.get(actual_columns.get('description', ''), '')),
                'part_number': str(row.get(actual_columns.get('part_number', ''), '')),
                'quantity': self._safe_float(row.get(actual_columns.get('quantity', ''), 0)),
                'unit_price': self._safe_float(row.get(actual_columns.get('unit_price', ''), 0)),
                'supplier_name': str(row.get(actual_columns.get('supplier_name', ''), '')),
                'category': str(row.get(actual_columns.get('category', ''), ''))
            }
            
            # Only add items with at least a description
            if item['description'].strip():
                items.append(item)

        return items

    def _safe_float(self, value) -> float:
        """Safely convert value to float"""
        try:
            if pd.isna(value):
                return 0.0
            return float(value)
        except:
            return 0.0

    def _create_demo_supplier_items(self) -> List[Dict]:
        """Create demo supplier items"""
        return [
            {
                'description': 'M6x20mm Stainless Steel Hex Bolt',
                'part_number': 'BOLT-M6-20-SS',
                'quantity': 100.0,
                'unit_price': 0.25,
                'supplier_name': 'FastenerCorp',
                'category': 'fasteners'
            },
            {
                'description': 'Industrial Adhesive Tape 25mm',
                'part_number': 'TAPE-ADH-25',
                'quantity': 50.0,
                'unit_price': 3.50,
                'supplier_name': 'AdhesivePlus',
                'category': 'adhesives'
            }
        ]

    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return self.stats.copy()