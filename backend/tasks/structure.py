from celery_app import celery_app
from ml.models.factory import build_model_from_json

@celery_app.task
def validate_model_structure(layer_json):
    try:
        model = build_model_from_json(layer_json)
        structure = []
        for idx, (name, module) in enumerate(model.named_children()):
            info = {
                "index": idx,
                "layer_id": getattr(module, "layer_id", None),
                "class_name": type(module).__name__,
                "repr": repr(module)
            }
            structure.append(info)
        return {"status": "success", "structure": structure}
    except Exception as e:
        return {"status": "error", "message": str(e)}
