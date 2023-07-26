import datetime
import time
import datetime as dt

BASE_START_DATE = "2015-01-01"

TICKER_LIST = {
    'GC=F' : 'com_disagg',
    '^GSPC': 'fut_fin',
    '^DJI' : '',
    '^IXIC' : '',
}
    
class Config(object):
    BASE_START_DATE = datetime.datetime.strptime(BASE_START_DATE, "%Y-%m-%d")
    JOBS = [
        # {
        #     'id': 'single_run',
        #     'func': 'scriptControllers.singleRunController:basicSingleRun',
        #     'kwargs': {
        #         'ticker' : 'GC=F',
        #         'mode' : 'com_disagg', 
        #         'start_date': BASE_START_DATE,
        #         'end_date': time.strftime("%Y-%m-%d", time.localtime())
        #         },
        #     'trigger': 'cron',
        #     'second': 30,
        # },
        {
            'id': 'multi',
            'func': 'scriptControllers.multiRunController:basicMultiRun',
            'kwargs': {
                'ticker_list' : TICKER_LIST,
                'start_date': BASE_START_DATE,
                'end_date': dt.datetime.today()
                },
            'trigger': 'cron',
            'second': 1,
        },
    ]
    SCHEDULER_API_ENABLED = True