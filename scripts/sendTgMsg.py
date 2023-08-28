import telegram
from telegram.request import HTTPXRequest
import os
import sys
import datetime
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.tgBotConfig import *
import asyncio
from utils.utils import *
from utils.getSignals import *
from utils.dict import *




class Bot():
    def __init__(self):
        self.timeout = 30
        # myRequests = HTTPXRequest(read_timeout=self.timeout, connect_timeout=self.timeout)
        self.bot = telegram.Bot(token=BOT_TOKEN) #, request=myRequests)
        self.chat_id = BETA_CHANNEL_ID
        print("Bot initialized. Request timeout: " + str(self.timeout) + "s.")
    async def greet(self):
        await self.sendMsg("Hello from the bot. today is " + time.strftime("%Y-%m-%d", time.localtime()) + ".")
        pass
    async def sendMsg(self, msg, parse_mode: str = 'Markdown'):
        await self.bot.send_message(chat_id=self.chat_id, text=msg, parse_mode=parse_mode)
        pass
    async def sendImg(self, img_path: str, caption: str = None):
        await self.bot.send_photo(chat_id=self.chat_id, photo=open(img_path, 'rb'), caption=caption)
        pass
    async def sendFile(self, file_path: str, caption: str = None):
        await self.bot.send_document(chat_id=self.chat_id, document=open(file_path, 'rb'))
        pass
    async def sendMediaGroup(self, media: list, caption: str = None, parse_mode: str = 'Markdown'):
        await self.bot.send_media_group(chat_id=self.chat_id, media=media, caption=caption, write_timeout = self.timeout,
                                        read_timeout = self.timeout, parse_mode = parse_mode)
        pass
    async def sendSticker(self, sticker_name: str):
        await self.bot.send_sticker(chat_id=self.chat_id, sticker=STICKERS[sticker_name])
        pass


def initBot()-> Bot:
    return Bot()

@retry_with_backoff(15)
async def send_daily_greetings(bot: Bot, date: str, tickers:list):
    await bot.sendMsg(DAILY_GREETING.format(date, str(tickers)))
    # print("Available tickers for " + date + ": " + str(tickers))
    # await bot.sendMsg("Available tickers for " + date + ": " + str(tickers))
    pass

@retry_with_backoff(15)
async def send_daily_ticker_report(bot: Bot, date: str, ticker: str, BtDict: dict, AIDict: dict, LogDict: dict):
    """Just send one gouped files and report for one ticker in one day.

    Args:
        bot (Bot): Tg bot
        date (str): date
        ticker (str): ticker name.
    """
    print("Sending daily report for " + ticker + " on " + date + "...")
    
    seperator = get_seperators()
    
    AI_file = AIDict[ticker][0] # By alphabetical order
    Bt_file = BtDict[ticker][1] # By alphabetical order
    today_signal, yesterday_close, bt_start_date, bt_end_date = get_signals(AI_file)
    # print("Today signal: " + str(today_signal) + ", yesterday close: " + str(yesterday_close) + ", bt start date: " + str(bt_start_date) + ", bt end date: " + str(bt_end_date))
    last_date, size, sharpeRatio, winRate, profitFactor, trades_per_year = get_trades(Bt_file)
    
    if last_date is not None:
        last_trade_msg = BT_STATS.format(bt_start_date, bt_end_date,last_date, winRate, profitFactor, trades_per_year, sharpeRatio)
    else:
        last_trade_msg = NO_LAST_TRADE
            
    if datetime.date.fromisoformat(bt_end_date) - datetime.date.fromisoformat(last_date) > datetime.timedelta(days=10):
        msg = NO_SIG_MSG.format(TICKER_DICT[ticker],date)
    else:
        msg = BUY_MSG.format(TICKER_DICT[ticker],yesterday_close, date) if '-' not in size else SELL_MSG.format(TICKER_DICT[ticker],yesterday_close, date)
    
    msg += last_trade_msg
    # print("Sending message: " + msg)
    media_group = []
    
    for file in BtDict[ticker] :
        if ".png" in file:
            media_group.append(telegram.InputMediaDocument(media=open(file, 'rb'), caption = msg, parse_mode='Markdown'))
    for file in AIDict[ticker] :
        if ".png" in file:
            media_group.append(telegram.InputMediaDocument(media=open(file, 'rb'), caption = 'AI signal'))
    for file in LogDict[ticker] :
        # print(file)
        if ".png" in file:
            filename = file.split(seperator)[-1].split('_')[-2] + '_' + file.split(seperator)[-1].split('_')[-1]
            media_group.append(telegram.InputMediaDocument(media=open(file, 'rb'), filename = filename
                                                           , caption = AI_CAPTION.format(file.split('_')[-1].split('.')[0])
                                                            , parse_mode='Markdown'))
    # print("Ready to send media group: " + str(media_group))
    await bot.sendMediaGroup(media_group, parse_mode='Markdown')
    
@retry_with_backoff(15)
async def conclude_daily_report(bot: Bot):
    await bot.sendMsg("Daily report finished.")
    await bot.sendSticker('bull')
    pass
    
def daily(date):
    """Forward daily report to telegram channel.
    Args:
        bot (Bot): Initialized Bot object.
        date (str): string of date in format YYYY-MM-DD.
    """
    # await bot.greet()
    bot = initBot()
    
    Log_paths, Bt_paths, AI_paths, tickers = get_available_tickers_and_paths(date)
    
    myAIDict = get_grouped_files(AI_paths, tickers)
    myBtDict = get_grouped_files(Bt_paths, tickers)
    myLogDict = get_grouped_files(Log_paths, tickers)
    # print(myLogDict)
    
    asyncio.run(send_daily_greetings(bot, date, list(TICKER_DICT.values())))
    
    for ticker in tickers:
        asyncio.run(send_daily_ticker_report(bot, date, ticker, myBtDict, myAIDict, myLogDict))
    
    asyncio.run(conclude_daily_report(bot))
    
    return "Daily task completed successfully"
    
if __name__ == '__main__':

    daily('2023-08-25')
    
    # bot = initBot()
    # asyncio.run(conclude_daily_report(bot))
    
    # loop.run_until_complete(daily('2023-07-27'))
    # files, tickers = get_available_tickers('2023-07-26')
    # print(get_grouped_files(files, tickers))
    
    pass
    
