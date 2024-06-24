import cv2
import numpy as np
import pyautogui
import pytesseract
import pyperclip  # 导入 pyperclip
import time
import os
import json
import re
from joblib import load

# 内部程序调用
from say import speak
from close import close_icons_main
from utils import scollscreen, capture_screen_area, predict_icon_status, load_model_and_scaler,find_icon
from model_config import models, templates, screen_regions


def main():
    """加载模型和标准化器"""
    clf_jump0, scaler_jump0 = models['jump0']
    template_jump0, w_jump0, h_jump0 = templates['jump0']
    
    clf_jump1, scaler_jump1 = models['jump1']
    template_jump1, w_jump1, h_jump1 = templates['jump1']

    clf_jump2, scaler_jump2 = models['jump2']
    template_jump2, w_jump2, h_jump2 = templates['jump2']

    clf_jump3, scaler_jump3 = models['jump3']
    template_jump3, w_jump3, h_jump3 = templates['jump3']

    clf_out1, scaler_out1 = models['out1']
    template_out1, w_out1, h_out1 = templates['out1']

    clf_zhongdian2, scaler_zhongdian2 = models['zhongdian2']
    template_zhongdian2, w_zhongdian2, h_zhongdian2 = templates['zhongdian2']

    # 屏幕区域配置
    region_full_right = screen_regions['full_right_panel']
    region_upper_right = screen_regions['upper_right_panel']
    mid_left_panel = screen_regions['mid_left_panel']


    """Start"""
    time.sleep(1)  # 等待开始
    close_icons_main()
    pyautogui.hotkey('ctrl', 'w')
    pyautogui.moveTo(450, 50)
    pyautogui.scroll(200)

    
    # 查找是否有[设置为终点]
    #""" 返程不需设置终点, 可注销这部分
    """attempts = 0
    max_attempts=5
    while attempts < max_attempts:        
        print('查找[设置终点]...')
        if find_icon(template_zhongdian2, w_zhongdian2, h_zhongdian2, clf_zhongdian2, scaler_zhongdian2, 1, 0, 0, mid_left_panel):
            pyautogui.leftClick()
            print('已设置终点!')
            break
        else:
            print('未找到[设置终点]!')
            attempts += 1
            scollscreen()
            time.sleep(0.5)
    #"""
    if find_icon(template_zhongdian2, w_zhongdian2, h_zhongdian2, clf_zhongdian2, scaler_zhongdian2, 5, 0, 0, mid_left_panel):
            pyautogui.leftClick()
            print('已设置终点!')

    # 持续查找小黄门
    print('查找[小黄门]...')
    while True : 
        if find_icon(template_jump0, w_jump0, h_jump0, clf_jump0, scaler_jump0,1,0,0,region_full_right):
            print('找到[小黄门]!')
            pyautogui.leftClick()            
            break
        # 尝试总览往下划动
        pyautogui.moveTo(1600,400)
        pyautogui.scroll(-900)
        print('未找到[小黄门],再次查找...')
    
    # 第一次跳跃
    print('第一次跳跃...')
    if find_icon(template_jump1, w_jump1, h_jump1, clf_jump1, scaler_jump1,1,0,0,region_full_right):
        pyautogui.leftClick()
        print('开始第一次跳跃...')

    time.sleep(0.5)

    print('第一次跃迁...')
    if find_icon(template_jump2, w_jump2, h_jump2, clf_jump2, scaler_jump2,1,0,0,region_full_right):
        pyautogui.leftClick()
        print('开始第一次跃迁...')
    
    #speak("欢迎登机, 您所乘坐的航班即将起飞,请收起小桌板,调整座椅靠背,手机设置飞行模式",185)
    time.sleep(3)

    """持续航行中"""
    while True:
        #关闭小窗口
        close_icons_main()

        # 持续检查是否可以停靠空间站
        if find_icon(template_jump3, w_jump3, h_jump3, clf_jump3, scaler_jump3,5,0,0,region_full_right):
            pyautogui.leftClick()            
            print('准备停靠空间站')
            # 只有停靠空间站才能退出循环
            break
        else:
            # 检查是否跳跃图标状态
            if find_icon(template_jump1, w_jump1, h_jump1, clf_jump1, scaler_jump1,1,0,0,region_full_right):
                pyautogui.leftClick()  
                print('跳跃至星门')
            else:
                # 如果跳跃不可用,则检查跃迁
                if find_icon(template_jump2, w_jump2, h_jump2, clf_jump2, scaler_jump2,1,0,0,region_full_right):
                    pyautogui.leftClick()
                    print('开始跃迁')
        
        # 重复检查
        time.sleep(2)

    # 停靠空间站
    while True:
        if find_icon(template_out1, w_out1, h_out1, clf_out1, scaler_out1,1,0,0,region_full_right):
            #speak('您的旅程已结束,感谢乘坐天合联盟东方航空,祝您旅途愉快',180)
            break
        time.sleep(1)

if __name__ == "__main__":
    main()


