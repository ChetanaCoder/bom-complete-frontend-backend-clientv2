"""
Autonomous Processing Router - FIXED VERSION
"""

import logging
import uuid
import asyncio
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from ..agents.agent_orchestrator import AgentOrchestrator
from ..utils.gemini_client import GeminiClient

logger = logging.getLogger(__name__)
router = APIRouter()

# Global variables (will be set by main.py)
orchestrator: Optional[AgentOrchestrator] = None
active_workflows = {}

def set_orchestrator(orch: AgentOrchestrator):
    """Set the orchestrator instance"""
    global orchestrator
    orchestrator = orch

async def update_workflow_progress(workflow_id: str, stage: str, progress: float, message: str):
    """Update workflow progress"""
    if workflow_id in active_workflows:
        active_workflows[workflow_id].update({
            "current_stage": stage,
            "progress": progress,
            "message": message,
            "status": "processing" if progress < 100 else "completed"
        })
        logger.info(f"Workflow {workflow_id}: {stage} - {progress}% - {message}")

@router.post("/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    wi_document: UploadFile = File(...),
    item_master: UploadFile = File(...)
):
    """Upload and process documents"""
    try:
        # Generate workflow ID
        workflow_id = str(uuid.uuid4())
        
        # Initialize workflow tracking
        active_workflows[workflow_id] = {
            "workflow_id": workflow_id,
            "status": "initializing",
            "current_stage": "upload",
            "progress": 0.0,
            "message": "Processing upload...",
            "wi_document": wi_document.filename,
            "item_master": item_master.filename
        }
        
        # Create uploads directory
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save uploaded files
        wi_path = upload_dir / wi_document.filename
        item_path = upload_dir / item_master.filename
        
        with open(wi_path, "wb") as f:
            content = await wi_document.read()
            f.write(content)
            
        with open(item_path, "wb") as f:
            content = await item_master.read()
            f.write(content)
        
        logger.info(f"Files uploaded for workflow {workflow_id}: {wi_document.filename}, {item_master.filename}")
        
        # Start background processing
        background_tasks.add_task(
            process_workflow_background,
            workflow_id,
            str(wi_path),
            str(item_path)
        )
        
        return JSONResponse({
            "success": True,
            "workflow_id": workflow_id,
            "message": "Documents uploaded successfully. Processing started.",
            "wi_document": wi_document.filename,
            "item_master": item_master.filename
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_workflow_background(workflow_id: str, wi_path: str, item_path: str):
    """Process workflow in background"""
    try:
        if not orchestrator:
            raise Exception("Orchestrator not initialized")
            
        # Update progress callback
        async def progress_callback(stage: str, progress: float, message: str):
            await update_workflow_progress(workflow_id, stage, progress, message)
        
        logger.info(f"Starting background processing for workflow {workflow_id}")
        
        # Process documents
        result = await orchestrator.process_documents_enhanced(
            wi_path,
            item_path, 
            workflow_id,
            progress_callback
        )
        
        # Update final status
        active_workflows[workflow_id].update({
            "status": "completed",
            "progress": 100.0,
            "message": "Processing completed successfully",
            "result": result
        })
        
        logger.info(f"Workflow {workflow_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Background processing failed for workflow {workflow_id}: {e}")
        active_workflows[workflow_id].update({
            "status": "error", 
            "progress": 0.0,
            "message": f"Processing failed: {str(e)}",
            "error": str(e)
        })

@router.get("/workflow/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get workflow processing status"""
    try:
        if workflow_id not in active_workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        status = active_workflows[workflow_id]
        return JSONResponse(status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflow/{workflow_id}/results")
async def get_workflow_results(workflow_id: str):
    """Get workflow results"""
    try:
        if workflow_id not in active_workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow = active_workflows[workflow_id]
        
        if workflow["status"] != "completed":
            raise HTTPException(status_code=400, detail="Workflow not completed yet")
        
        if "result" not in workflow:
            raise HTTPException(status_code=404, detail="Results not found")
        
        return JSONResponse(workflow["result"])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Results retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows")
async def list_workflows():
    """List all workflows"""
    try:
        workflows = []
        for workflow_id, workflow in active_workflows.items():
            workflows.append({
                "workflow_id": workflow_id,
                "status": workflow.get("status", "unknown"),
                "progress": workflow.get("progress", 0.0),
                "message": workflow.get("message", ""),
                "wi_document": workflow.get("wi_document", ""),
                "item_master": workflow.get("item_master", "")
            })
        
        return JSONResponse({"workflows": workflows})
        
    except Exception as e:
        logger.error(f"Workflow listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
