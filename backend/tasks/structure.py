import os
import json
import uuid
from celery_app import celery_app
from ml.models.factory import build_model_from_json

@celery_app.task
def validate_model_structure(layer_json):
    try:
        if not isinstance(layer_json, list):
            return {"status": "error", "message": "layer_json must be a list of layer configs"}

        model = build_model_from_json(layer_json)
        structure = []
        for idx, (name, module) in enumerate(model.named_children()):
            r = repr(module)
            # if len(r) > 500:  # 너무 길면 잘라서 전달
            #     r = r[:500] + " ... (truncated)"
            info = {
                "index": idx,
                "layer_id": getattr(module, "layer_id", None),
                "class_name": type(module).__name__,
                "repr": r,
            }
            structure.append(info)

        # 파라미터 수 요약 (디버깅 편의)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # --- 전체 구조를 JSON 파일로 저장 ---
        os.makedirs("/app/backend/model_structures", exist_ok=True)  # 볼륨 마운트 권장
        file_id = str(uuid.uuid4())[:8]
        filepath = f"/app/backend/model_structures/model_structure_{file_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(structure, f, indent=2, ensure_ascii=False)
        print(f"Model structure saved to {filepath}")

        return {
            "status": "success",
            "structure": structure,
            "summary": {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
            },
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
