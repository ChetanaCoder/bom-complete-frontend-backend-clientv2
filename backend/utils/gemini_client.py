"""
Enhanced Gemini AI Client with real API integration
"""

import logging
import os
import json
import asyncio
import aiohttp
from typing import Optional

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.available = bool(self.api_key)
        
        if not self.available:
            logger.warning("Gemini API key not found - running in demo mode")
        else:
            logger.info("Gemini client initialized successfully")
            # Set the API key as environment variable for use
            os.environ['GEMINI_API_KEY'] = self.api_key

    def is_available(self) -> bool:
        """Check if Gemini client is available"""
        return self.available

    async def generate_content(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Generate content using Gemini API"""
        if not self.is_available():
            logger.warning("Gemini API not available, returning demo response")
            return self._get_demo_response(prompt)
        
        try:
            url = f"{self.base_url}?key={self.api_key}"
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "topP": 0.8,
                    "topK": 10
                }
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extract text from Gemini response
                        if 'candidates' in result and len(result['candidates']) > 0:
                            candidate = result['candidates'][0]
                            if 'content' in candidate and 'parts' in candidate['content']:
                                parts = candidate['content']['parts']
                                if len(parts) > 0 and 'text' in parts[0]:
                                    generated_text = parts[0]['text'].strip()
                                    logger.info(f"Successfully generated {len(generated_text)} characters from Gemini")
                                    return generated_text
                        
                        logger.warning("Unexpected response format from Gemini API")
                        return self._get_demo_response(prompt)
                    else:
                        error_text = await response.text()
                        logger.error(f"Gemini API error {response.status}: {error_text}")
                        return self._get_demo_response(prompt)
                        
        except asyncio.TimeoutError:
            logger.error("Gemini API request timed out")
            return self._get_demo_response(prompt)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._get_demo_response(prompt)

    def _get_demo_response(self, prompt: str) -> str:
        """Generate demo response when API is not available"""
        if "translate" in prompt.lower():
            return "Demo English translation: This is a technical work instruction document containing assembly steps, material specifications, and quality control procedures."
        elif "extract materials" in prompt.lower():
            demo_json = [
                {
                    "name": "M6x20mm Hex Bolt",
                    "category": "fasteners",
                    "specifications": {"size": "M6x20mm", "type": "hex bolt", "material": "stainless steel"},
                    "context": "Use M6×20 hex bolts for chassis assembly in step 3",
                    "confidence_score": 0.95,
                    "qc_process_step": "Assembly Step 3",
                    "consumable_jigs_tools": True,
                    "name_mismatch": False,
                    "part_number": "BOLT-M6-20-SS",
                    "pn_mismatch": False,
                    "quantity": 4,
                    "unit_of_measure": "pieces",
                    "obsolete_pn": False,
                    "vendor_name": "FastenerCorp",
                    "kit_available": False,
                    "ai_engine_processing": "Demo mode - AI classification"
                }
            ]
            return json.dumps(demo_json, indent=2)
        else:
            return f"Demo AI response for: {prompt[:100]}..."
