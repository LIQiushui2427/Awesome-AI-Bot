"""Should somehow store the mapping from ticker to CFTC code.
"""
STICKERS = {
    'bull': 'CAACAgUAAxkBAAEkHnhkv5Ws_JNni65Arl4tPy09kHq5wQACjwYAAlAoSFRhUoGOAAHwZ84vBA',
    'bear': 'CAACAgUAAxkBAAEkHu9kv57O543EZGPYoyM01xY2cMmLMAACAQcAAjqoSFTA9PIEg5aXuS8E',
}

TICKER_LIST = {
    'GC=F' : 'com_disagg',
    '^GSPC': 'fut_fin',
    '^DJI' : '',
    '^IXIC' : '',
    '0388.HK' : '',
    '^HSI' : '',
}
TICKER_DICT = {
    'GC=F': '黃金期貨到期日12月23日',
    '^DJI': '道瓊工業平均指數',
    '^IXIC': '納斯達克綜合指數',
    '^GSPC': '標普500指數',
    '0388.HK': '港交所',
    '^HSI': '恆生指數',
}