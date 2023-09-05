from __future__ import absolute_import, unicode_literals
from flask import Flask
from myCelery import make_celery, daily
from flask_apscheduler import APScheduler
import Configs.celeryConfig as config
from Configs.APSConfig import Config
from scripts.sendTgMsg import *

flask_app = Flask(__name__)

flask_app.config.from_object(Config())
scheduler = APScheduler()
scheduler.init_app(flask_app)
scheduler.start()


# celery = make_celery(flask_app)


if __name__ == "__main__":
    flask_app.run(port=config.PORT, debug=False)
