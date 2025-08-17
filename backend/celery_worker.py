from celery_app import celery_app
import tasks.train
import tasks.structure

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
