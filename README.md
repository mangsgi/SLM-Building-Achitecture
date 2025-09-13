# Building SLM Achitecture

### 1. backend: fastapi 기본 test 코드

```bash
cd backend
python -m venv venv
source venv/bin/activate # if powershell, .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app
```

### 2. frontend: react 데모 layer 웹

```bash
cd frontend
npm install
npm run dev
```

- 현재 eslint, prettier, husky 적용 중, 언어는 TypeScript 사용

### 통합

```bash
# 루트에서
docker compose up -d --build

# 모든 로그 보기
docker compose logs -f

# 컨테이너 중지(네트워크/볼륨은 유지)
docker compose stop

# 컨테이너 + 네트워크 삭제(볼륨/이미지는 유지)
docker compose down

# 컨테이너 + 네트워크 + 볼륨까지 삭제(실험 데이터도 지워짐 주의)
docker compose down -v
```

- 패키징 시 Llama-2 토크나이저로 인해 라이선스 파일을 함께 포함(licenses/LLAMA-2-LICENSE.txt) (라이선스 전문 동봉)
- NOTICE 문구 유지 (파일명은 관례상 NOTICE 또는 NOTICE.txt)
  - Llama 2 is licensed under the LLAMA 2 Community License,
  - Copyright (c) Meta Platforms, Inc. All Rights Reserved.
