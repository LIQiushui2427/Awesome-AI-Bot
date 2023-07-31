import telegram
from telegram.request import HTTPXRequest
import os
import sys
import time
import re

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.tgBotConfig import *
import asyncio
from utils.utils import *
from utils.getSignals import *

STICKERS = {
    'bull': 'CAACAgUAAxkBAAEkHnhkv5Ws_JNni65Arl4tPy09kHq5wQACjwYAAlAoSFRhUoGOAAHwZ84vBA',
    'bear': 'CAACAgUAAxkBAAEkHu9kv57O543EZGPYoyM01xY2cMmLMAACAQcAAjqoSFTA9PIEg5aXuS8E',
}

TICKER_DICT = {
    'GC=F': 'Gold',
    '^DJI': 'Dow Jones',
    '^IXIC': 'Nasdaq',
    '^GSPC': 'S&P 500',
}


class Bot():
    def __init__(self):
        self.timeout = 30
        # myRequests = HTTPXRequest(read_timeout=self.timeout, connect_timeout=self.timeout)
        self.bot = telegram.Bot(token=BOT_TOKEN) #, request=myRequests)
        self.chat_id = CHANNEL_ID
        print("Bot initialized. Request timeout: " + str(self.timeout) + "s.")
    async def greet(self):
        await self.sendMsg("Hello from the bot. today is " + time.strftime("%Y-%m-%d", time.localtime()) + ".")
    async def sendMsg(self, msg):
        await self.bot.send_message(chat_id=self.chat_id, text=msg)
    async def sendImg(self, img_path: str, caption: str = None):
        await self.bot.send_photo(chat_id=self.chat_id, photo=open(img_path, 'rb'), caption=caption)
    async def sendFile(self, file_path: str, caption: str = None):
        await self.bot.send_document(chat_id=self.chat_id, document=open(file_path, 'rb'))
    async def sendMediaGroup(self, media: list, caption: str = None, parse_mode: str = 'HTML'):
        await self.bot.send_media_group(chat_id=self.chat_id, media=media, caption=caption, write_timeout = self.timeout,
                                        read_timeout = self.timeout, parse_mode = parse_mode)
    async def sendSticker(self, sticker_name: str):
        await self.bot.send_sticker(chat_id=self.chat_id, sticker=STICKERS[sticker_name])


def initBot()-> Bot:
    return Bot()

@retry_with_backoff(15)
async def send_daily_greetings(bot: Bot, date: str, tickers:list):
    await bot.sendMsg(DAILY_GREETING.format(date))
    # print("Available tickers for " + date + ": " + str(tickers))
    await bot.sendMsg("Available tickers for " + date + ": " + str(tickers))

@retry_with_backoff(15)
async def send_daily_ticker_report(bot: Bot, date: str, ticker: str, BtDict: dict, AIDict: dict, LogDict: dict):
    """Just send one gouped files and report for one ticker in one day.

    Args:
        bot (Bot): Tg bot
        date (str): date
        ticker (str): ticker name.
    """
    print("Sending daily report for " + ticker + " on " + date + "...")
    
    media_group = []
    for file in BtDict[ticker] :
        if ".png" in file:
            media_group.append(telegram.InputMediaDocument(media=open(file, 'rb')))
        elif ".txt" in file:
            media_group.append(telegram.InputMediaDocument(media=open(file, 'rb')))
    for file in AIDict[ticker] :
        if ".png" in file:
            media_group.append(telegram.InputMediaDocument(media=open(file, 'rb')))
        elif ".txt" in file:
            media_group.append(telegram.InputMediaDocument(media=open(file, 'rb')))
    for file in LogDict[ticker] :
        # print(file)
        if ".png" in file:
            media_group.append(telegram.InputMediaDocument(media=open(file, 'rb')))
        elif ".log" in file:
            media_group.append(telegram.InputMediaDocument(media=open(file, 'rb')))
            
    AI_file = AIDict[ticker][0] # By alphabetical order
    Bt_file = BtDict[ticker][1] # By alphabetical order

    today_signal, yesterday_close, bt_start_date, bt_end_date = get_signals(AI_file)
    last_date, size = get_trades(Bt_file)
    
    if last_date is not None:
        last_trade_msg = LAST_TRADE.format(last_date)
    else:
        last_trade_msg = NO_LAST_TRADE
            
    msg = MSG_HTML.format(date, TICKER_DICT[ticker]) + BUY_MSG.format(yesterday_close) if today_signal > 0 else SELL_MSG.format(yesterday_close)
    
    caption = msg + last_trade_msg + CAPTION.format(bt_start_date, bt_end_date)
    
    # print("Ready to send media group: " + str(media_group))
    
    await bot.sendMediaGroup(media_group, caption, parse_mode='Markdown')
    
@retry_with_backoff(15)
async def conclude_daily_report(bot: Bot):
    await bot.sendMsg("Daily report finished.")
    await bot.sendSticker('bull')
    
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
    
    asyncio.run(send_daily_greetings(bot, date, tickers))
    
    for ticker in tickers:
        asyncio.run(send_daily_ticker_report(bot, date, ticker, myBtDict, myAIDict, myLogDict))
    
    asyncio.run(conclude_daily_report(bot))
    
    return "Daily task completed successfully"
    
if __name__ == '__main__':

    # daily('2023-07-31')
    
    # bot = initBot()
    # asyncio.run(conclude_daily_report(bot))
    
    # loop.run_until_complete(daily('2023-07-27'))
    # files, tickers = get_available_tickers('2023-07-26')
    # print(get_grouped_files(files, tickers))
    
    pass
    
