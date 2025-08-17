# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.train_routes import router as train_router
from routes.inference_routes import router as inference_router
from routes.completed_routes import router as completed_router
from routes.events import router as events_router
from routes.stop_routes import router as stop_router  # ✅ 추가

app = FastAPI(title="SLM Model Builder",
              description="SLM 모델 구조 생성 및 확인 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록 (공통 prefix는 여기서만)
app.include_router(train_router, prefix="/api/v1")
app.include_router(inference_router, prefix="/api/v1")
app.include_router(completed_router, prefix="/api/v1")
app.include_router(events_router, prefix="/api/v1")
app.include_router(stop_router, prefix="/api/v1")  # ✅ 추가

@app.get("/")
def root():
    return {"message": "SLM Trainer is running!"}
