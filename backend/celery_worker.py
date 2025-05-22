from celery import Celery


celery_app = Celery(
    'worker',
    broker="redis://localhost:6379/0",
    result_backend="redis://localhost:6379/0",
    include=['tasks']  # tasks 모듈을 직접 포함
)

celery_app.conf.update(
    result_expires=3600,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=True,
    worker_concurrency=1,
    worker_pool='solo',
)

# tasks 모듈 자동 검색 비활성화
# celery_app.autodiscover_tasks(['tasks'])