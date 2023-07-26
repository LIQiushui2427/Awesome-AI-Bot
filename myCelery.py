from __future__ import absolute_import, unicode_literals
from celery import Celery
from flask import Flask
from Configs import celeryConfig

import time
from scripts.sendTgMsg import *

TICKER_LIST = {
    'GC=F' : 'fut_disagg',
    '^GSPC': '',
    '^DJI' : '',
    '^IXIC' : '',
}



def make_celery(app : Flask):
    
    celery = Celery(celery_name = celeryConfig.celery_name,
                broker=celeryConfig.BROKER_URL,
                backend=celeryConfig.BACKEND_URL)
    
    # celery.config_from_object('Configs.celeryConfig')
    celery.conf.timezone = celeryConfig.CELERY_TIMEZONE
    celery.conf.enable_utc = celeryConfig.CELERY_ENABLE_UTC
    
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    app.celery = celery

    # # 添加任务
    # celery.task(name="send_daily")(daily)

    return celery