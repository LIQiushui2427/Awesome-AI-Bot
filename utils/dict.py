# -*- coding: UTF-8 -*-
"""Should somehow store the mapping from ticker to CFTC code.
"""
STICKERS = {
    "bull": "CAACAgUAAxkBAAEkHnhkv5Ws_JNni65Arl4tPy09kHq5wQACjwYAAlAoSFRhUoGOAAHwZ84vBA",
    "bear": "CAACAgUAAxkBAAEkHu9kv57O543EZGPYoyM01xY2cMmLMAACAQcAAjqoSFTA9PIEg5aXuS8E",
}

TICKER_LIST = {
    "^GSPC": "fut_fin",
    "^DJI": "",
    "^IXIC": "",
    "^HSI": "",
    "^HSCE": "",
    "000001.SS": "",
    "0388.HK": "",
    "FUTU": "",
    "BTC-USD": "",
    "NVDA": "",
    "INTC": "",
    "TSLA": "",
    "AAPL": "",
    "BILI": "",
    "AMD": "",
    "AMZN": "",
    "GOOG": "",
}
TICKER_DICT = {
    "GC=F": "#黃金期貨到期日12月23日",
    "^DJI": "#道瓊工業平均指數",
    "^IXIC": "#納斯達克綜合指數",
    "^GSPC": "#標普500指數",
    "NVDA": "#英伟达",
    "FUTU": "#富途控股",
    "000001.SS": "#上证指数",
    "0388.HK": "#港交所",
    "^HSI": "#恆生指數",
    "BTC-USD": "#比特幣对美元",
    "^HSCE": "#恒生中國企業指數",
    "TSLA": "#特斯拉",
    "AAPL": "#蘋果",
    "BILI": "#哔哩哔哩",
    "INTC": "#英特尔",
    "AMD": "#AMD",
    "AMZN": "#亞馬遜",
    "CL=F": "#WTI原油期貨到期日9月23日",
    "GOOG": "#谷歌",
    "UNIBOT27009-USD": "#UNIBOT",
}

MAPPING = {
    "^GSPC": "13874+",
    "GC=F": "001602",
    "CL=F": "06765C",
}
