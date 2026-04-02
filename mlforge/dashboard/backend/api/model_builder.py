"""Model Builder API — design, generate, and register custom neural networks."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from dashboard.backend.models.db import CustomModel, get_session
from dashboard.backend.services.model_codegen import (
    LAYER_CATALOG,
    TEMPLATES,
    estimate_params,
    explain_layer,
    generate_pytorch_code,
    register_custom_model,
)

logger = logging.getLogger("mlforge.api.model_builder")

router = APIRouter(prefix="/api/model-builder", tags=["model-builder"])


class GenerateRequest(BaseModel):
    layers: list[dict]
    model_name: str = "CustomModel"
    num_classes: int = 10


class RegisterRequest(BaseModel):
    name: str
    description: str = ""
    layers: list[dict]
    num_classes: int = 10


class TemplateRequest(BaseModel):
    template_id: str
    params: dict = {}


# ─── Layer Catalog ──────────────────────────────────────────────────────

@router.get("/layers")
def get_layer_catalog():
    """Return all available layer types with params and explanations."""
    result = {}
    for name, info in LAYER_CATALOG.items():
        result[name] = {
            "category": info["category"],
            "params": info["params"],
            "explanation": info["explanation"],
            "tip": info.get("tip", ""),
        }
    logger.debug("[MODEL-BUILDER] Layer catalog: %d layers", len(result))
    return result


@router.get("/layers/{layer_type}/explain")
def get_layer_explanation(layer_type: str):
    """Get detailed educational explanation for a layer type."""
    info = explain_layer(layer_type)
    if info["category"] == "Unknown":
        raise HTTPException(status_code=404, detail=f"Unknown layer type: {layer_type}")
    return info


# ─── Templates ──────────────────────────────────────────────────────────

@router.get("/templates")
def list_templates():
    """List available architecture templates."""
    result = []
    for tid, t in TEMPLATES.items():
        result.append({
            "id": tid,
            "name": t["name"],
            "description": t["description"],
            "difficulty": t["difficulty"],
            "params": {k: {**v} for k, v in t["params"].items()},  # copy without generate fn
        })
    return result


@router.post("/templates/apply")
def apply_template(data: TemplateRequest):
    """Apply a template with given params and return the generated layers."""
    template = TEMPLATES.get(data.template_id)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template not found: {data.template_id}")

    layers = template["generate"](data.params)
    logger.info("[MODEL-BUILDER] Applied template '%s' with params=%s -> %d layers", data.template_id, data.params, len(layers))
    return {"layers": layers, "template": data.template_id}


# ─── Code Generation ────────────────────────────────────────────────────

@router.post("/generate")
def generate_code(data: GenerateRequest):
    """Generate PyTorch code from a list of layer definitions."""
    logger.info("[MODEL-BUILDER] Generate code: %s, %d layers, %d classes", data.model_name, len(data.layers), data.num_classes)

    code = generate_pytorch_code(data.layers, data.model_name, data.num_classes)
    num_params = estimate_params(data.layers, data.num_classes)

    # Generate per-layer explanations
    explanations = []
    for layer in data.layers:
        lt = layer["type"]
        info = LAYER_CATALOG.get(lt, {})
        explanations.append({
            "type": lt,
            "explanation": info.get("explanation", ""),
            "tip": info.get("tip", ""),
        })

    return {
        "code": code,
        "num_params": num_params,
        "num_params_human": f"{num_params / 1e6:.2f}M" if num_params >= 1e6 else f"{num_params / 1e3:.1f}K",
        "num_layers": len(data.layers),
        "explanations": explanations,
    }


# ─── Custom Models CRUD ─────────────────────────────────────────────────

@router.get("/models")
def list_custom_models(session: Session = Depends(get_session)):
    """List all saved custom models."""
    models = session.exec(select(CustomModel).order_by(CustomModel.created_at.desc())).all()
    logger.debug("[MODEL-BUILDER] Listed %d custom models", len(models))
    return models


@router.post("/models", status_code=201)
def save_and_register_model(data: RegisterRequest, session: Session = Depends(get_session)):
    """Save a custom model to DB and register it for training."""
    logger.info("[MODEL-BUILDER] Saving model '%s' (%d layers)", data.name, len(data.layers))

    # Check name uniqueness
    existing = session.exec(select(CustomModel).where(CustomModel.name == data.name)).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Model name '{data.name}' already exists")

    # Generate code
    safe_class_name = "".join(w.capitalize() for w in data.name.replace("-", "_").split("_"))
    code = generate_pytorch_code(data.layers, safe_class_name, data.num_classes)
    num_params = estimate_params(data.layers, data.num_classes)

    # Try to register dynamically
    success = register_custom_model(data.name, code, data.num_classes)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to register model. Check the architecture is valid.")

    # Save to DB
    model = CustomModel(
        name=data.name,
        description=data.description,
        layers_json=json.dumps(data.layers),
        code=code,
        num_params=num_params,
    )
    session.add(model)
    session.commit()
    session.refresh(model)

    logger.info("[MODEL-BUILDER] Model '%s' saved (id=%d, params=%d) and registered for training", data.name, model.id, num_params)
    return model


@router.delete("/models/{model_id}", status_code=204)
def delete_custom_model(model_id: int, session: Session = Depends(get_session)):
    """Delete a custom model."""
    model = session.get(CustomModel, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Remove from registry
    from mlforge.models.registry import _PYTORCH_REGISTRY
    if model.name in _PYTORCH_REGISTRY:
        del _PYTORCH_REGISTRY[model.name]

    session.delete(model)
    session.commit()
    logger.info("[MODEL-BUILDER] Deleted model '%s' (id=%d)", model.name, model_id)
