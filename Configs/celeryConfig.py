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
        # 具体需要执行的函数
        # 该函数必须要使用@app.task装饰
        'task': 'myCelery.app_scripts.test1.test1_run',
        # 定时时间
        # 每分钟执行一次，不能为小数
        'schedule': crontab(minute='*/1'),
        # 或者这么写，每小时执行一次
        # "schedule": crontab(minute=0, hour="*/1")
        # 执行的函数需要的参数
        'args': ()
    },
    # 'testdaily': {
    #     'task': 'celery_task.app_scripts.test2.test2_run',
    #     # 设置定时的时间，10秒一次
    #     'schedule': timedelta(seconds=10),
    #     'args': ()
    # }
}

task_serializer = 'pickle'

result_serializer = 'pickle'

accept_content = ['pickle', 'json']