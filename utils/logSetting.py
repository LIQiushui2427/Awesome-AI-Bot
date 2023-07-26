# -*-coding:utf-8-*-
# 设置格式
import logging
import os


def get_log(file_name):
    logger = logging.getLogger('AItrain')  # 设定logger的名字
    logger.setLevel(logging.INFO)  # 设定logger得等级

    fh = logging.FileHandler(file_name, mode='w')  # 文件流的hander，输出得文件名称，以及mode设置为覆盖模式
    fh.setLevel(logging.INFO)  # 设定文件hander得lever



    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # 将hander添加到我们声明的logger中去
    
    return logger