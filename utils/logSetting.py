# -*-coding:utf-8-*-
#
import logging
import os


def get_log(file_name):
    logger = logging.getLogger("AItrain")  # logger
    logger.setLevel(logging.INFO)  # logger

    fh = logging.FileHandler(file_name, mode="w")  # hander，，mode
    fh.setLevel(logging.INFO)  # handerlever

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # handerlogger

    return logger
