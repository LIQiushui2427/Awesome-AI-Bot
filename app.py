"""
# coding:utf-8
@Time    : 2023/07/15
@Author  : Easy
@File    : app.py
@Software: Vscode
"""

from flask import Flask, render_template
import threading
from flask_apscheduler import APScheduler
import time
from APSConfigs.AIConfig import *
import logging


app = Flask(__name__)
app.config.from_object(Config())
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()
# logging.getLogger('apscheduler.executors.default').setLevel(logging.WARNING)

@app.route("/")
def index():
    return 'ok'

if __name__ == '__main__':
    app.run(port=8000,debug=False)
