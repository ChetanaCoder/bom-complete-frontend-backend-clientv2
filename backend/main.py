"""
FastAPI Backend for BOM Analysis System - FIXED VERSION
"""

import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .utils.gemini_client import GeminiClient
from .agents.agent_orchestrator import AgentOrchestrator
from .routers import autonomous, knowledge_base

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
gemini_client = None
orchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global gemini_client, orchestrator
    try:
        # Set the Gemini API key
        gemini_api_key = "AIzaSyAmpbcqcghKn-IjkhBvdJU-LLfaf4JFEx4"
        os.environ['GEMINI_API_KEY'] = gemini_api_key
        
        # Initialize Gemini client with the API key
        gemini_client = GeminiClient(api_key=gemini_api_key)
        
        # Initialize orchestrator
        orchestrator = AgentOrchestrator(gemini_client)
        
        # Set orchestrator in router
        autonomous.set_orchestrator(orchestrator)
        
        logger.info("Backend initialized successfully")
        logger.info(f"Gemini client available: {gemini_client.is_available()}")
        
    except Exception as e:
        logger.error(f"Failed to initialize backend: {e}")
        # Continue anyway for debugging
    
    yield
    
    # Shutdown
    logger.info("Backend shutting down")

app = FastAPI(
    title="BOM Analysis API",
    description="Enhanced BOM Analysis with QA Classification - FIXED VERSION",
    version="2.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(autonomous.router, prefix="/api/autonomous", tags=["autonomous"])
app.include_router(knowledge_base.router, prefix="/api/knowledge-base", tags=["knowledge"])

@app.get("/")
async def root():
    return {
        "message": "BOM Analysis API v2.1 - Enhanced with QA Classification",
        "status": "running",
        "gemini_available": gemini_client.is_available() if gemini_client else False
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.1.0",
        "gemini_available": gemini_client.is_available() if gemini_client else False,
        "orchestrator_ready": orchestrator is not None,
        "api_key_configured": bool(os.getenv('GEMINI_API_KEY'))
    }

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify basic functionality"""
    try:
        return {
            "message": "Test successful",
            "gemini_client": gemini_client is not None,
            "gemini_available": gemini_client.is_available() if gemini_client else False,
            "orchestrator": orchestrator is not None,
            "api_key_length": len(os.getenv('GEMINI_API_KEY', '')) if os.getenv('GEMINI_API_KEY') else 0
        }
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        return {"error": str(e)}
