import cv2
import numpy as np
import pyautogui
import pytesseract
import pyperclip  # 导入 pyperclip
import time
from joblib import load
import os
import json
from cnocr import CnOcr
import re

# 内部程序调用
from say import speak
from close import close_icons_main
from outsite import outsite_icons_main
from utils import scollscreen, capture_screen_area, predict_icon_status, load_model_and_scaler,find_icon,load_location_name,find_txt_ocr,find_txt_ocr2
from model_config import models, templates, screen_regions



def navigate_main():
    clf_kjz1, scaler_kjz1 = models['kjz1']
    template_kjz1, w_kjz1, h_kjz1 = templates['kjz1']

    clf_search1, scaler_search1 = models['search1']
    template_search1, w_search1, h_search1 = templates['search1']

    clf_zhongdian1, scaler_zhongdian1 = models['zhongdian1']
    template_zhongdian1, w_zhongdian1, h_zhongdian1 = templates['zhongdian1']



    # 执行关闭窗口操作
    time.sleep(2)
    close_icons_main()

    # 1. 按 'l' 键打开地点搜索窗口
    pyautogui.press('l')
    
    # 2. 检测并点击 "搜索"
    if find_icon(template_search1, w_search1, h_search1, clf_search1, scaler_search1,2,22,0):
        pyautogui.leftClick()

        # 3. 输入地点名称并回车
        location_name = load_location_name('addr')
        if location_name:
            pyperclip.copy(location_name)  # 复制地点名称到剪贴板
            pyautogui.leftClick() 
            time.sleep(1)
            pyautogui.hotkey('ctrl', 'v')  # 粘贴地点名称
            pyautogui.press('enter')
            time.sleep(1)  # 等待搜索结果
    
        # 4. 检测并点击空间站图标
        if find_icon(template_kjz1, w_kjz1, h_kjz1, clf_kjz1, scaler_kjz1,2,0,22) :
            pyautogui.rightClick()
            time.sleep(0.5) 
            # 5. 检测并点击 "设定为终点"
            if find_icon(template_zhongdian1, w_zhongdian1, h_zhongdian1, clf_zhongdian1, scaler_zhongdian1,2):
                time.sleep(0.5) 
                pyautogui.leftClick()
                time.sleep(0.5) 
                print("导航已设定!")

    close_icons_main()
    pyautogui.hotkey('ctrl', 'w')
    #出站
    time.sleep(2) 
    outsite_icons_main()
   


if __name__ == "__main__":
    navigate_main()