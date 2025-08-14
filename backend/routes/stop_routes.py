# routes/stop_routes.py
from __future__ import annotations

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from deps import rds_sync, train_event_channel
from celery_app import celery_app
from celery.result import AsyncResult

# prefix 제거: main.py에서 /api/v1 붙임
router = APIRouter(tags=["training-stop"])

# ----------------------------
# Redis Key Helpers
# ----------------------------
def _stop_key(task_id: str) -> str:
    return f"train:stop:{task_id}"

def _set_stop_flag(task_id: str) -> None:
    rds_sync.set(_stop_key(task_id), "1")

def _clear_stop_flag(task_id: str) -> None:
    rds_sync.delete(_stop_key(task_id))

def _get_stop_status(task_id: str) -> dict:
    flag = rds_sync.get(_stop_key(task_id))
    return {
        "stopping": (flag == "1" or flag == b"1"),
    }

def _publish_event(task_id: str, event: str, data: dict) -> None:
    try:
        payload = {"event": event, "data": data}
        rds_sync.publish(train_event_channel(task_id), json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass

# ----------------------------
# Request Models
# ----------------------------
class StopRequest(BaseModel):
    task_id: str
    force_kill: bool = False

class ResumeRequest(BaseModel):
    task_id: str

# ----------------------------
# Routes
# ----------------------------
@router.post("/stop-training")
def stop_training(req: StopRequest):
    task_id = (req.task_id or "").strip()
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")

    res = AsyncResult(task_id, app=celery_app)
    state = res.state

    _set_stop_flag(task_id)
    _publish_event(task_id, "stop_requested", {"task_id": task_id, "state": state})

    killed = False
    if req.force_kill and state not in ("SUCCESS", "FAILURE", "REVOKED"):
        try:
            res.revoke(terminate=True, signal="SIGTERM")
            killed = True
            _publish_event(task_id, "stop_forced", {"task_id": task_id})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Force kill failed: {e}")

    return {
        "status": "ok",
        "task_id": task_id,
        "celery_state": state,
        "mode": "force_kill" if req.force_kill else "cooperative_stop",
        "message": "Stop requested (flag set). Hard-kill executed." if killed else "Stop requested (flag set).",
    }
