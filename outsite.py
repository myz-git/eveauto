# outsite.py
import pyautogui
import time
from utils import safe_find_icon, screen_regions,find_txt_ocr

def outsite_icons_main():
    # 使用全屏区域或指定区域
    region = screen_regions['full_right_panel']
    agent_panel3 = screen_regions['agent_panel3']
        
    # 关闭导航窗口
    if find_txt_ocr("关闭",max_attempts=1,region=agent_panel3):
        time.sleep(0.2) 
        pyautogui.leftClick()

    # 查找并点击 out1 图标
    # if safe_find_icon("out1", region, max_attempts=5,cnn_threshold=0.7):
    if find_txt_ocr("离站",max_attempts=5,region=region):
        time.sleep(0.2) 
        pyautogui.leftClick()
        time.sleep(3)
        outsite_check()

def outsite_check():
    # 使用全屏区域或指定区域
    region = screen_regions['full_right_panel']

    # 判断是否出站
    attempts = 0
    while attempts < 20:
        if safe_find_icon("zonglan1", region, max_attempts=10, offset_x=0, offset_y=-100,cnn_threshold=0.8):
            print("已出站,程序退出")
            break 
        time.sleep(0.5)
        attempts += 1

if __name__ == "__main__":
    outsite_icons_main()