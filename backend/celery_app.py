from celery import Celery
import os

# 환경 변수에서 브로커 및 백엔드 URL을 가져오고, 없을 경우 기본값을 사용합니다.
broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend_url = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery(
    "worker",
    broker=broker_url,
    backend=result_backend_url,
)

# Celery 상세 설정
celery_app.conf.update(
    # Redis 브로커의 가시성 시간 초과(visibility timeout)를 300초(5분)로 설정합니다.
    # 이는 작업이 300초 안에 완료 신호(ack)를 보내지 않으면 브로커가 작업을 다른 워커에게 다시 할당하는 것을 방지합니다.
    broker_transport_options={
        'visibility_timeout': 300
    }
)