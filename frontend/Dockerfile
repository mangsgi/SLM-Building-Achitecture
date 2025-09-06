FROM node:20-alpine

WORKDIR /app/frontend

# 재빌드 속도를 위해 메타파일만 먼저 복사 → npm ci
COPY package*.json ./
RUN npm ci

# 실제 소스 복사 (개발 중엔 볼륨이 덮어씀)
COPY . .

EXPOSE 5173
CMD ["npm","run","dev","--","--host","0.0.0.0"]
