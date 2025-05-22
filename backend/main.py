from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import train_routes

app = FastAPI(title="SLM Model Builder",
             description="SLM 모델 구조 생성 및 확인 API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(train_routes.router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "SLM Trainer is running!"}
