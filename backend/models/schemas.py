"""
Enhanced BOM Analysis Models and Schemas - FIXED VERSION
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Any
from enum import IntEnum, Enum
from datetime import datetime

class QAClassificationLabel(IntEnum):
    """QA Classification labels (1-13) based on processing rules"""
    CONSUMABLE_WITH_PN_SPEC_QTY = 1
    CONSUMABLE_WITH_PN_QTY = 2
    CONSUMABLE_NO_QTY = 3
    CONSUMABLE_NO_PN = 4
    CONSUMABLE_OBSOLETE_PN = 5
    CONSUMABLE_PN_MISMATCH = 6
    VENDOR_KIT_NO_PN = 7
    VENDOR_NAME_ONLY = 8
    PRE_ASSEMBLED_KIT = 9
    NO_CONSUMABLE_MENTIONED = 10
    JIGS_TOOLS_IDENTIFIED = 11
    PARTIAL_INFO_AVAILABLE = 12
    REQUIRES_MANUAL_REVIEW = 13

class ConfidenceLevel(str, Enum):
    """Confidence levels for QA processing"""
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"

class ActionPathRAG(str, Enum):
    """Action path based on RAG (Red, Amber, Green) classification"""
    GREEN = "green"   # Auto-register
    AMBER = "amber"   # Auto with flag  
    RED = "red"       # Human intervention required

class ExtractedMaterial(BaseModel):
    """Enhanced extracted material with QA classification - FIXED VERSION"""
    # Original fields
    name: str = Field(..., description="Material name/description")
    qa_material_name: Optional[str] = Field(None, description="QA material name (alias for name)")
    category: str = Field(default="uncategorized", description="Material category")
    specifications: Dict[str, Any] = Field(default_factory=dict, description="Material specifications")
    context: str = Field(default="", description="Context from document")
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Extraction confidence")
    source_section: str = Field(default="", description="Source document section")
    qa_excerpt: Optional[str] = Field(None, description="Excerpt from QA document")

    # Enhanced QA fields
    qc_process_step: Optional[str] = Field(None, description="QC process step if mentioned")
    consumable_jigs_tools: bool = Field(default=False, description="Is consumable/jigs/tools")
    name_mismatch: bool = Field(default=False, description="Name mismatch detected")
    part_number: Optional[str] = Field(None, description="Part number if available")
    pn_mismatch: bool = Field(default=False, description="Part number mismatch")
    quantity: Optional[Union[float, int]] = Field(None, description="Quantity if specified")
    unit_of_measure: Optional[str] = Field(None, description="Unit of measure")
    obsolete_pn: bool = Field(default=False, description="Obsolete part number")
    vendor_name: Optional[str] = Field(None, description="Vendor name if mentioned")
    kit_available: bool = Field(default=False, description="Kit availability")
    ai_engine_processing: str = Field(default="AI processed", description="AI processing notes")

    # Classification results - FIXED: Handle both enum and int values
    confidence_level: Union[ConfidenceLevel, str] = Field(default=ConfidenceLevel.MEDIUM)
    action_path_rag: Union[ActionPathRAG, str] = Field(default=ActionPathRAG.AMBER)
    classification_label: Union[QAClassificationLabel, int] = Field(default=QAClassificationLabel.REQUIRES_MANUAL_REVIEW)
    classification_reasoning: str = Field(default="", description="Reasoning for classification")
    
    # Additional fields for compatibility
    qa_classification_label: Optional[int] = Field(None, description="QA classification label as int")
    qa_confidence_level: Optional[str] = Field(None, description="QA confidence level as string")

    @validator('qa_material_name', pre=True, always=True)
    def set_qa_material_name(cls, v, values):
        """Ensure qa_material_name is set from name if not provided"""
        if v is None and 'name' in values:
            return values['name']
        return v or ""

    @validator('qa_classification_label', pre=True, always=True)
    def set_qa_classification_label(cls, v, values):
        """Set qa_classification_label from classification_label"""
        if v is None and 'classification_label' in values:
            label = values['classification_label']
            # FIXED: Handle both enum and int values
            if isinstance(label, QAClassificationLabel):
                return label.value
            elif isinstance(label, int):
                return label
            else:
                return QAClassificationLabel.REQUIRES_MANUAL_REVIEW.value
        return v or QAClassificationLabel.REQUIRES_MANUAL_REVIEW.value

    @validator('qa_confidence_level', pre=True, always=True)
    def set_qa_confidence_level(cls, v, values):
        """Set qa_confidence_level from confidence_level"""
        if v is None and 'confidence_level' in values:
            level = values['confidence_level']
            if isinstance(level, ConfidenceLevel):
                return level.value
            elif isinstance(level, str):
                return level
            else:
                return ConfidenceLevel.MEDIUM.value
        return v or ConfidenceLevel.MEDIUM.value

    class Config:
        use_enum_values = True
        # Allow extra fields for flexibility
        extra = "allow"

class SupplierBOMItem(BaseModel):
    """Supplier BOM item model"""
    description: str = Field(..., description="Item description")
    part_number: str = Field(default="", description="Supplier part number")
    quantity: Optional[Union[float, int]] = Field(None, description="Quantity")
    unit_price: Optional[float] = Field(None, description="Unit price")
    category: str = Field(default="", description="Item category")
    supplier_name: str = Field(default="", description="Supplier name")

class MaterialMatch(BaseModel):
    """Enhanced material match result"""
    # Original material fields
    qa_material_name: str = Field(..., description="QA material name")
    qa_excerpt: Optional[str] = Field(None, description="QA excerpt")
    qc_process_step: Optional[str] = Field(None, description="QC process step")
    part_number: Optional[str] = Field(None, description="Part number")
    
    # Classification fields
    qa_classification_label: int = Field(..., description="QA classification label")
    qa_confidence_level: str = Field(..., description="QA confidence level")
    
    # Match information
    confidence_score: float = Field(default=0.0, description="Match confidence score")
    supplier_description: str = Field(default="", description="Supplier description")
    supplier_part_number: str = Field(default="", description="Supplier part number")
    match_source: str = Field(default="no_match", description="Source of match")
    reasoning: str = Field(default="", description="Match reasoning")
    has_previous_match: bool = Field(default=False, description="Has knowledge base match")
    
    # Additional fields
    category: str = Field(default="", description="Material category")
    specifications: Dict[str, Any] = Field(default_factory=dict)
    context: str = Field(default="", description="Context")

    class Config:
        extra = "allow"  # Allow extra fields

class QAClassificationSummary(BaseModel):
    """QA Classification summary"""
    total_materials: int = Field(default=0)
    green_materials: int = Field(default=0)
    amber_materials: int = Field(default=0)
    red_materials: int = Field(default=0)
    classification_breakdown: Dict[str, int] = Field(default_factory=dict)

class BOMComparisonResult(BaseModel):
    """BOM comparison result model"""
    workflow_id: str = Field(..., description="Workflow identifier")
    matches: List[Union[MaterialMatch, Dict]] = Field(default_factory=list, description="Material matches")
    summary: Dict = Field(default_factory=dict, description="Comparison summary")

class WorkflowStatus(BaseModel):
    """Workflow status model"""
    workflow_id: str
    status: str = Field(default="initializing")
    current_stage: str = Field(default="starting")
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    message: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
