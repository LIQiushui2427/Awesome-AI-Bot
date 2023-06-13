import re
import requests
from bs4 import BeautifulSoup
import importlib

import sys
importlib.reload(sys)

# %2C%22priceToBookRatio%22%3A1.1390819712409195%2C
# r'%2C%22priceToBookRatio%22%3A(.*?)%2C'

url='https://www.bloomberg.com/quote/HSI:IND'

headers = { 'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36' }
    
html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text.encode('utf-8')

f = open('result.txt','w',encoding='utf-8')

f.write(html.decode('utf-8'))

regex = r'%2C%22priceToBookRatio%22%3A(.*?)%2C'

result = re.findall(regex, html.decode('utf-8'))

print(result[0])