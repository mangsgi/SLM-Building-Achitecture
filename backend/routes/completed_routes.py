from fastapi import APIRouter
from pathlib import Path

router = APIRouter()

@router.get("/completed-models", tags=["Completed Models"])
def list_completed_models():
    completed_dir = Path("completed")
    structures_dir = Path("temp_structures")

    pt_models = {f.stem for f in completed_dir.glob("*.pt")}
    pth_models = {f.stem for f in completed_dir.glob("*.pth")}
    bin_models = {f.stem for f in completed_dir.glob("*.bin")}
    completed_models = sorted(list(pt_models.union(pth_models).union(bin_models)))
    structure_files = {f.stem: f.name for f in structures_dir.glob("*.json")}

    result = []
    for model_name in completed_models:
        result.append({
            "model_name": model_name,
            "has_structure": model_name in structure_files,
            "structure_file": structure_files.get(model_name)
        })

    return {"models": result}
