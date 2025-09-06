# Python 3.11.8 (README 요구사항 상위 호환)
FROM python:3.11.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# 컨텍스트가 ./backend 이므로 파일명만 사용
COPY requirements.txt /tmp/requirements.txt

# 한 번에 설치 (requirements.txt에 uvicorn[standard] 반영 권장)
RUN pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt

# 소스 복사 (개발 중엔 compose 볼륨이 덮어씀)
COPY . /app/backend

EXPOSE 8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
