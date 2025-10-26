"""
API de ejemplo para servir el modelo de simplificación de textos médicos.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from .model import MedicalSimplifier
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="API de Simplificación de Textos Médicos",
    description="API para simplificar textos médicos técnicos en explicaciones claras para pacientes",
    version="1.0.0"
)

# Modelos Pydantic para la API
class SimplificationRequest(BaseModel):
    text: str
    max_length: Optional[int] = 128
    num_beams: Optional[int] = 4

class BatchSimplificationRequest(BaseModel):
    texts: List[str]
    batch_size: Optional[int] = 8
    max_length: Optional[int] = 128
    num_beams: Optional[int] = 4

class SimplificationResponse(BaseModel):
    original_text: str
    simplified_text: str

class BatchSimplificationResponse(BaseModel):
    results: List[SimplificationResponse]

# Instancia global del modelo
model = None

@app.on_event("startup")
async def load_model():
    """Carga el modelo al iniciar la aplicación"""
    global model
    try:
        model = MedicalSimplifier(use_half_precision=True)
        logger.info("Modelo cargado exitosamente")
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
        raise

@app.post("/simplify", response_model=SimplificationResponse)
async def simplify_text(request: SimplificationRequest):
    """
    Simplifica un texto médico.
    
    Args:
        request: Texto médico y parámetros opcionales
        
    Returns:
        SimplificationResponse: Texto original y simplificado
    """
    try:
        simplified = model.simplify(
            request.text,
            max_length=request.max_length,
            num_beams=request.num_beams
        )
        return SimplificationResponse(
            original_text=request.text,
            simplified_text=simplified
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_simplify", response_model=BatchSimplificationResponse)
async def batch_simplify_texts(request: BatchSimplificationRequest):
    """
    Simplifica múltiples textos médicos en lote.
    
    Args:
        request: Lista de textos y parámetros opcionales
        
    Returns:
        BatchSimplificationResponse: Lista de textos originales y simplificados
    """
    try:
        simplified_texts = model.batch_simplify(
            request.texts,
            batch_size=request.batch_size,
            max_length=request.max_length,
            num_beams=request.num_beams
        )
        
        results = [
            SimplificationResponse(original_text=orig, simplified_text=simp)
            for orig, simp in zip(request.texts, simplified_texts)
        ]
        
        return BatchSimplificationResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Verifica el estado de la API y el modelo.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available()
    }