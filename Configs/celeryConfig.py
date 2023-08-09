from celery.schedules import crontab
from datetime import timedelta
import time
celery_name = "app"

backend_url = 'redis://127.0.0.1:6379/0'
broker_url = 'redis://127.0.0.1:6379/0'

PORT = 3000

timezone = "Asia/Shanghai"
enable_utc = False

result_expires = 60 * 60 * 24

beat_schedule = {
    'daily': {
        # 
        # @app.task
        'task': 'myCelery.app_scripts.test1.test1_run',
        # 
        # ，
        'schedule': crontab(minute='*/1'),
        # ，
        # "schedule": crontab(minute=0, hour="*/1")
        # 
        'args': ()
    },
    # 'testdaily': {
    #     'task': 'celery_task.app_scripts.test2.test2_run',
    #     # ，10
    #     'schedule': timedelta(seconds=10),
    #     'args': ()
    # }
}

task_serializer = 'pickle'

result_serializer = 'pickle'

accept_content = ['pickle', 'json']