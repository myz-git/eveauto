# wk1.py
import numpy as np
import pyautogui
import pyperclip
import time
from joblib import load
from cnocr import CnOcr
import re
import sys
import logging
import pynput
from utils import scrollscreen, capture_screen_area, safe_find_icon, find_txt_ocr, find_txt_ocr2, correct_string, screen_regions, close_icons_main, log_message


def compress():
    ctr = pynput.keyboard.Controller()
    with ctr.pressed(pynput.keyboard.Key.ctrl, 'p'):
        time.sleep(0.2)
        pass
    time.sleep(0.2)

def wk1_check():
    attempts = 0
    while True:
        if compress():
            print("已压缩")
        time.sleep(10)
if __name__ == "__main__":
    wk1_check()