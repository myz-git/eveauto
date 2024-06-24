# outsite.py
import cv2
import numpy as np
import pyautogui
import time
from joblib import load
import os
from model_config import models, templates
from utils import capture_screen_area,find_icon


def outsite_icons_main():
    # Load models and scalers
    clf_out1, scaler_out1 = models['out1']
    template_out1, w_out1, h_out1 = templates['out1']

    if find_icon(template_out1, w_out1, h_out1, clf_out1, scaler_out1,3):
        pyautogui.leftClick()
        outsite_check()

def outsite_check():
    clf_zonglan1, scaler_zonglan1 = models['zonglan1']
    template_zonglan1, w_zonglan1, h_zonglan1 = templates['zonglan1']
    # 判断是否出站了
    attempts = 0
    while attempts < 20:
        if find_icon(template_zonglan1, w_zonglan1, h_zonglan1, clf_zonglan1, scaler_zonglan1,1,0,-100) :   
            print("已出站,程序退出")
            break 
        time.sleep(0.5)
        attempts += 1        
        

if __name__ == "__main__":
    outsite_icons_main()
