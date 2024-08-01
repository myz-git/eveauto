# close.py
import cv2
import numpy as np
import pyautogui
import time
from joblib import load
import os
from model_config import models, templates
from utils import capture_screen_area


def find_and_close_icons(template, clf, scaler, w, h):
    
    fx, fy = pyautogui.size()
    region = (0, 0, fx, fy)
    screen = capture_screen_area(region)

    #res = cv2.matchTemplate(gray_screen, template_gray, cv2.TM_CCOEFF_NORMED)
    res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED) 
    threshold = 0.8
    loc = np.where(res >= threshold)

    if loc[0].size > 0:
        for pt in zip(*loc[::-1]):  # Switch columns and rows
            global_x = pt[0] + w // 2
            global_y = pt[1] + h // 2
            pyautogui.moveTo(global_x+11, global_y)
            pyautogui.click()
            time.sleep(1) 
            #print(f"Clicked close icon at: ({global_x}, {global_y})")
        return True
    return False

def close_icons_main():
    # Load models and scalers"""加载模型和标准化器"""
    clf_close1, scaler_close1 = models['close1']
    template_close1, w_close1, h_close1 = templates['close1']
    w_close1, h_close1 = template_close1.shape[1], template_close1.shape[0]  # 修改宽高的获取方式

    find_and_close_icons(template_close1, clf_close1, scaler_close1, w_close1, h_close1)

if __name__ == "__main__":
    close_icons_main()
