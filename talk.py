import numpy as np
import pyautogui
import time
from joblib import load
from cnocr import CnOcr

# 内部程序调用
from say import speak
from close import close_icons_main
from utils import scollscreen, capture_screen_area, predict_icon_status, load_model_and_scaler,find_icon,load_location_name,find_txt_ocr,find_txt_ocr2
from model_config import models, templates, screen_regions
from navigate import navigate_main

class IconNotFoundException(Exception):
    """Exception raised when an icon is not found."""
    pass

class GoodsNotFoundException(Exception):
    """Exception raised when the specified goods are not found."""
    pass


def main():

    clf_talk1, scaler_talk1 = models['talk1']
    template_talk1, w_talk1, h_talk1 = templates['talk1']

    clf_talk2, scaler_talk2 = models['talk2']
    template_talk2, w_talk2, h_talk2 = templates['talk2']

    clf_agent1, scaler_agent1 = models['agent1']
    template_agent1, w_agent1, h_agent1 = templates['agent1']
    

    # 设置需要捕获的屏幕区域
    fx,fy=pyautogui.size()
    
    #左侧面板(左中)
    mid_left_panel = screen_regions['mid_left_panel']
    #中间对话框
    agent_panel3 = screen_regions['agent_panel3']

    #设置需要捕获的屏幕区域
    agent_panel1 = screen_regions['agent_panel1']
    #代理人列表窗口
    agent_panel2 = screen_regions['agent_panel2']


    # 1. 准备开始
    
    # 查找[开始对话]
    if find_icon(template_talk1, w_talk1, h_talk1, clf_talk1, scaler_talk1,3,0,0,mid_left_panel):
        pyautogui.leftClick()        
        print("开始对话...")
        time.sleep(1)
    else:
         # 2. 查找"代理人"图标  
        try:  
            if find_icon(template_agent1, w_agent1, h_agent1, clf_agent1, scaler_agent1,2,0,0,agent_panel1):
                pyautogui.leftClick()
        except IconNotFoundException as e:
            print(e)
    
        # 3. 查找代理人
        # 3.1 获得代理人名字
        agent_name = load_location_name('agent')
        print(f"agent={agent_name}")

        # 3.2 通过OCR文字识别查找代理人
        if find_txt_ocr(agent_name,1,agent_panel2):
            pyautogui.hotkey('ctrl', 'w')
            pyautogui.doubleClick()  # 双击打开代理人对话窗口
            time.sleep(0.5)

    # 4. 和代理人开始对话
    # 查找[目标完成]
    if find_txt_ocr("目标完成",3,agent_panel3):
        # 当找到[目标完成],查找[完成任务了]并点击;
        if find_icon(template_talk2, w_talk2, h_talk2, clf_talk2, scaler_talk2,3,0,0,agent_panel3):
            pyautogui.leftClick()
            print("完成任务了...")
            
    time.sleep(0.5)
    print("任务完成,准备返航")
    # 2. 设定返回导航 并出站
    ##navigate_main()
    #print("返航出站...")
    
    
if __name__ == "__main__":
    main()
