import telegram
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

class Bot():
    def __init__(self):
        self.bot = telegram.Bot(token=BOT_TOKEN)
        self.chat_id = CHANNEL_ID
        print("Bot initialized.")
    async def greet(self):
        await self.sendMsg("Hello from the bot. today is " + time.strftime("%Y-%m-%d", time.localtime()) + ".")
    async def sendMsg(self, msg):
        await self.bot.send_message(chat_id=self.chat_id, text=msg)
    async def sendImg(self, img_path: str, caption: str = None):
        await self.bot.send_photo(chat_id=self.chat_id, photo=open(img_path, 'rb'), caption=caption)
    async def sendFile(self, file_path: str, caption: str = None):
        await self.bot.send_document(chat_id=self.chat_id, document=open(file_path, 'rb'))
    async def sendMediaGroup(self, media: list, caption: str = None):
        await self.bot.send_media_group(chat_id=self.chat_id, media=media, caption=caption)
    async def sendSticker(self, sticker_name: str):
        await self.bot.send_sticker(chat_id=self.chat_id, sticker=STICKERS[sticker_name])

def initBot()-> Bot:
    return Bot()


        
async def daily(date):
    """Forward daily report to telegram channel.
    Args:
        bot (Bot): Initialized Bot object.
        date (str): string of date in format YYYY-MM-DD.
    """
    # await bot.greet()
    
    bot = initBot()
    await bot.sendMsg("Sending daily report for " + date + "...")
    files, tickers = get_available_tickers(date)
    await bot.sendMsg("Available tickers for " + date + ": " + str(tickers))
    
    
    # send grouped files
    mySet = get_grouped_files(files, tickers)
    for ticker in mySet.keys():
        media_group = []
        for file in mySet[ticker]:
            media_group.append(telegram.InputMediaDocument(media=open(file, 'rb')))
        
        AI_file = os.path.join(os.getcwd(), 'outputsByAI', f'{ticker}.csv')
        
        today_signal, yesterday_close = get_signals(AI_file)
        
        msg = BUY_MSG.format(date, ticker, yesterday_close) if today_signal > 0 else SELL_MSG.format(date, ticker, yesterday_close)
        await bot.sendMediaGroup(media_group, caption= msg)
        
    await bot.sendSticker('bull')
    
    return "Daily task completed successfully"
    
if __name__ == '__main__':
    asyncio.run(daily('2023-07-26'))
    # files, tickers = get_available_tickers('2023-07-25')
    # print(get_grouped_files(files, tickers))
