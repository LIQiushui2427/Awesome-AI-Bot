import datetime
import time
class Config(object):
    BASE_START_DATE = "2015-01-01"
    JOBS = [
        {
            'id': 'fetchData',
            'func': 'scriptControllers.singleRunController:basicSingleRun',
            'kwargs': {
                'ticker' : 'GC=F',
                 'mode' : 'Com_Disagg', 
                 'start_date': BASE_START_DATE,
                 'end_date': time.strftime("%Y-%m-%d", time.localtime())
                },
            'trigger': 'interval',
            'seconds': 60
        },
    ]
    SCHEDULER_API_ENABLED = True