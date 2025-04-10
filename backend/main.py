from fastapi import FastAPI
from routes import train_routes  

app = FastAPI()

# 라우터 등록 (prefix는 API 버전 또는 기능별로만 설정하는 게 일반적)
app.include_router(train_routes.train_router)

@app.get("/")
def root():
    return {"message": "SLM Trainer is running!"}
