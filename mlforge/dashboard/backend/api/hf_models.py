"""HuggingFace model hub API — browse, download, and register pretrained models."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from dashboard.backend.models.db import HFModel, get_session
from dashboard.backend.services.hf_hub import (
    download_and_register,
    list_popular_models,
    search_models,
)

logger = logging.getLogger("mlforge.api.hf_models")

router = APIRouter(prefix="/api/hf-models", tags=["hf-models"])


class DownloadRequest(BaseModel):
    timm_name: str
    num_classes: int = 10


@router.get("/popular")
def get_popular():
    """List curated popular pretrained models."""
    models = list_popular_models()
    logger.debug("[HF-MODELS] Returning %d popular models", len(models))
    return models


@router.get("/search")
def search(q: str = "", limit: int = 20):
    """Search timm models by name."""
    if not q or len(q) < 2:
        return []
    results = search_models(q, limit=limit)
    logger.info("[HF-MODELS] Search '%s': %d results", q, len(results))
    return results


@router.get("/downloaded")
def list_downloaded(session: Session = Depends(get_session)):
    """List all downloaded/registered HF models."""
    models = session.exec(select(HFModel).order_by(HFModel.downloaded_at.desc())).all()
    return models


@router.post("/download", status_code=201)
def download_model(data: DownloadRequest, session: Session = Depends(get_session)):
    """Download a timm model and register it for training/export."""
    logger.info("[HF-MODELS] Download request: %s (classes=%d)", data.timm_name, data.num_classes)

    # Check if already downloaded
    existing = session.exec(
        select(HFModel).where(HFModel.timm_name == data.timm_name)
    ).first()
    if existing:
        logger.info("[HF-MODELS] Model '%s' already downloaded (id=%d)", data.timm_name, existing.id)
        return existing

    try:
        result = download_and_register(data.timm_name, data.num_classes)
    except ImportError:
        logger.error("[HF-MODELS] timm not installed")
        raise HTTPException(status_code=500, detail="timm library not installed. Rebuild the Docker image.")
    except Exception as e:
        logger.exception("[HF-MODELS] Download failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to download model: {e}")

    # Find display info from curated list
    from dashboard.backend.services.hf_hub import POPULAR_MODELS
    curated = next((m for m in POPULAR_MODELS if m["timm_name"] == data.timm_name), None)

    hf_model = HFModel(
        timm_name=data.timm_name,
        display_name=curated["display_name"] if curated else data.timm_name.replace("_", " ").title(),
        registry_name=result["registry_name"],
        num_params=result["num_params"],
        default_input_size=result["input_size"],
    )
    session.add(hf_model)
    session.commit()
    session.refresh(hf_model)

    logger.info(
        "[HF-MODELS] Downloaded and registered: %s -> %s (%.2fM params)",
        data.timm_name, result["registry_name"], result["num_params"] / 1e6,
    )
    return hf_model


@router.delete("/downloaded/{model_id}", status_code=204)
def remove_downloaded(model_id: int, session: Session = Depends(get_session)):
    """Remove a downloaded model from the registry."""
    model = session.get(HFModel, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Remove from registry
    from mlforge.models.registry import _PYTORCH_REGISTRY
    if model.registry_name in _PYTORCH_REGISTRY:
        del _PYTORCH_REGISTRY[model.registry_name]

    session.delete(model)
    session.commit()
    logger.info("[HF-MODELS] Removed model '%s' (id=%d)", model.timm_name, model_id)
