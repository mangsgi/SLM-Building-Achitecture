from fastapi import APIRouter, HTTPException
from celery.result import AsyncResult
from celery_worker import celery_app

router = APIRouter()

@router.post("/stop/{task_id}")
def stop_training(task_id: str):
    try:
        result = AsyncResult(task_id, app=celery_app)
        if result.state == 'PENDING':
            return {"status": "error", "message": "Task not found or not started yet"}
        celery_app.control.revoke(task_id, terminate=True)
        return {"status": "success", "message": f"Training task {task_id} has been stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 