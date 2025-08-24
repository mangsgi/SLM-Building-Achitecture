# 📌 SLM Model Builder Backend

JSON 기반으로 **소형 언어 모델(SLM)** 구조를 설계하고, 학습 및 추론까지 수행하는 **FastAPI + Celery + Redis + MLflow** 백엔드입니다.

---

## 📦 요구사항

- **Python 3.10+**
- **Redis 서버** (`redis-server`)
- **PyTorch** (CPU 또는 CUDA 환경에 맞게 설치)
- **MLflow** (실험 및 로그 관리)
- (선택) **GPU 환경 권장**

---

## 📥 설치 방법

```bash
git clone <this-repo-url>
cd <this-repo-name>

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

```

## 실행 순서

```bash
# 1) 가상환경 + 설치
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) MLflow UI
mlflow ui --host 127.0.0.1 --port 5000

# 3) Redis
redis-server
# 필요 시 초기화: redis-cli flushall

# 4) Celery Worker
celery -A celery_worker.celery_app worker --loglevel=info

# 5) FastAPI
uvicorn main:app --reload
```

## 📥 API 사용법

---

### 1️⃣ 학습 시작 — `POST /api/v1/train-complete-model`

모델 구조 검증 → 구조 저장 → MLflow 런 생성 → Celery 학습 태스크 시작 → SSE로 진행상황 스트리밍

```bash
curl -X POST http://127.0.0.1:8000/api/v1/train-complete-model \
  -H "Content-Type: application/json" \
  -d @request.json

  request.json 예시

  "config": {
    "model": "gpt-2",
    "epochs": 1,
    "batch_size": 2,
    "vocab_size": 50257,
    "context_length": 32,
    "emb_dim": 128,
    "n_heads": 2,
    "n_blocks": 1,
    "drop_rate": 0.1,
    "qkv_bias": false,
    "dtype": "fp32"
  },
  "model": [
    { "type": "tokenEmbedding", "data": { "id": "tok", "vocabSize": 50257, "embDim": 128 } },
    { "type": "positionalEmbedding", "data": { "id": "pos", "ctxLength": 32, "embDim": 128, "mode": "learned" } },
    { "type": "linear", "data": { "id": "head", "inDim": 128, "outDim": 50257 } }
  ],
  "dataset": "tiny_shakespeare",
  "dataset_config": "default",
  "modelName": "ex1"
}

✅ 성공 응답 예시

json
{
  "status": "success",
  "task_id": "abc123",
  "sse_url": "/api/v1/events/abc123",
  "sse_url_abs": "http://127.0.0.1:8000/api/v1/events/abc123",
  "stop_url_abs": "http://127.0.0.1:8000/api/v1/stop-training",
  "mlflow_url": "http://127.0.0.1:5000/#/experiments/1/runs/run123",
  "message": "모델 검증/생성/구조저장/MLflow 런 생성 후 학습을 시작했습니다. 진행상황은 SSE로 전송됩니다."
}
```

2️⃣ 진행상황 스트리밍 — GET /api/v1/events/{task_id}
학습 이벤트(connected, started, step, progress, status, finished, error)를 실시간 푸시
15초마다 하트비트(: keep-alive) 전송

```bash


curl -N http://127.0.0.1:8000/api/v1/events/<TASK_ID>


data: {"event":"connected","data":{"channel":"task:<TASK_ID>"}}
data: {"event":"started","data":{"task_id":"<TASK_ID>","model_name":"ex1","run_id":"<RUN_ID>","mlflow_url":"http://127.0.0.1:5000/#/experiments/1/runs/<RUN_ID>","epochs":1,"batch_size":2,"context_length":32,"stride":16}}
data: {"event":"step","data":{"task_id":"<TASK_ID>","global_step":10,"epoch":1,"loss":2.3451,"ema_loss":2.2103}}
data: {"event":"progress","data":{"task_id":"<TASK_ID>","epoch":1,"epochs":1,"avg_epoch_loss":2.10,"train_loss":2.05,"val_loss":2.40,"global_step":120,"run_id":"<RUN_ID>"}}
data: {"event":"status","data":{"task_id":"<TASK_ID>","state":"running","epoch":1,"global_step":120}}
data: {"event":"finished","data":{"task_id":"<TASK_ID>","model_name":"ex1","run_id":"<RUN_ID>","completed_model_path":"completed/ex1.pt","last_train_loss":2.05,"last_val_loss":2.40,"status":"finished","ts":1720000000.0}}
```

3️⃣ 학습 중단 — POST /api/v1/stop-training

협력적(cooperative) 중단이 기본, force_kill=true 시 하드 종료(SIGTERM)

```bash

curl -X POST http://127.0.0.1:8000/api/v1/stop-training \
  -H "Content-Type: application/json" \
  -d '{"task_id":"<TASK_ID>", "force_kill": false}'


✅ 응답 예시

{
  "status": "ok",
  "task_id": "<TASK_ID>",
  "celery_state": "STARTED",
  "mode": "cooperative_stop",
  "message": "Stop requested (flag set)."
}

```

4️⃣ 완료 모델 목록 — GET /api/v1/completed-models

```bash
curl http://127.0.0.1:8000/api/v1/completed-models


✅ 응답 예시

{
  "models": [
    { "model_name": "ex1", "has_structure": true,  "structure_file": "ex1.json" },
    { "model_name": "ex2", "has_structure": false, "structure_file": null }
  ]
}

```

5️⃣ 텍스트 생성(추론) — POST /api/v1/generate-text

```bash
curl -X POST http://127.0.0.1:8000/api/v1/generate-text \
  -H "Content-Type: application/json" \
  -d '{"model_name":"ex1","input_text":"Once upon a time","max_length":50,"temperature":0.7,"top_k":40}'


✅ 응답 예시

{
  "status": "success",
  "input_text": "Once upon a time",
  "output_text": "Once upon a time ... (모델이 생성한 텍스트)"
}

```
