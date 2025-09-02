"""
Knowledge Base Router - FIXED VERSION
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..database.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize knowledge base
try:
    kb = KnowledgeBase()
except Exception as e:
    logger.error(f"Failed to initialize knowledge base: {e}")
    kb = None

@router.get("/stats")
async def get_knowledge_base_stats():
    """Get knowledge base statistics"""
    try:
        if not kb:
            raise HTTPException(status_code=503, detail="Knowledge base not available")
        
        stats = kb.get_processing_stats()
        return JSONResponse(stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stats retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear")
async def clear_knowledge_base():
    """Clear all data from knowledge base"""
    try:
        if not kb:
            raise HTTPException(status_code=503, detail="Knowledge base not available")
        
        kb.clear_all_data()
        return JSONResponse({"success": True, "message": "Knowledge base cleared successfully"})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clear operation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/items")
async def search_items(query: str = "", limit: int = 50):
    """Search items in knowledge base"""
    try:
        if not kb:
            raise HTTPException(status_code=503, detail="Knowledge base not available")
        
        # For now, return empty results - can be enhanced later
        return JSONResponse({
            "items": [],
            "total": 0,
            "query": query,
            "limit": limit
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
