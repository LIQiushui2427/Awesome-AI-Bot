from __future__ import absolute_import, unicode_literals
from flask import Flask, jsonify, current_app
from myCelery import make_celery, daily
from flask_apscheduler import APScheduler
import Configs.celeryConfig as config
from Configs.APSConfig import Config
from scripts.sendTgMsg import *
import asyncio
import time

flask_app = Flask(__name__)

# celery = make_celery(flask_app)

flask_app.config.from_object(Config())
scheduler = APScheduler()
scheduler.init_app(flask_app)
scheduler.start()

# celery = make_celery(flask_app)


# @celery.task(name="send_daily")
# async def send_daily(date):
#     daily(date)

# send_daily.apply_async(args=[time.strftime("%Y-%m-%d", time.localtime())])

# celery.task(name="run")(send_daily)
# celery.send_task('send_daily', args=(time.strftime("%Y-%m-%d", time.localtime()),))

@flask_app.route('/', methods=['GET'])
def hello_world():
    return jsonify({'msg': 'Hello World!'})

@flask_app.route('/send_daily', methods=['GET'])
def send_daily():
    try:
        current_app.celery.send_task('send_daily', args=(time.strftime("%Y-%m-%d", time.localtime())))
        return jsonify({'msg': 'send_daily task sent!'})
    except Exception as e:
        return jsonify({'msg': 'send_daily task failed!'})
    
if __name__ == '__main__':
    flask_app.run(port=config.PORT, debug=False)