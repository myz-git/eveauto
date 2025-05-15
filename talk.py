# talk.py
import pyautogui
import time
import pynput

# 内部程序调用
from say import speak
from utils import scollscreen, capture_screen_area, safe_find_icon, load_location_name, find_txt_ocr, find_txt_ocr2, screen_regions,close_icons_main
from navigate import navigate_main

class GoodsNotFoundException(Exception):
    """Exception raised when the specified goods are not found."""
    pass

def main():
    # 设置需要捕获的屏幕区域
    fx, fy = pyautogui.size()
    
    # 左侧面板(左中)
    mid_left_panel = screen_regions['mid_left_panel']
    # 中间对话框
    agent_panel3 = screen_regions['agent_panel3']
    # 设置需要捕获的屏幕区域
    agent_panel1 = screen_regions['agent_panel1']
    # 代理人列表窗口
    agent_panel2 = screen_regions['agent_panel2']

    ctr = pynput.keyboard.Controller()
    # 1. 准备开始
    
    # 查找[开始对话]
    if safe_find_icon("talk1", mid_left_panel, max_attempts=3):
        pyautogui.leftClick()        
        print("开始对话...")
        time.sleep(1)
    else:
        # 2. 查找"代理人"图标  
        if safe_find_icon("agent1", agent_panel1, max_attempts=2):
            pyautogui.leftClick()
    
        # 3. 查找代理人
        # 3.1 获得代理人名字
        agent_name = load_location_name('agent')
        print(f"agent={agent_name}")

        # 3.2 通过OCR文字识别查找代理人
        if find_txt_ocr(agent_name, 1, agent_panel2):
            with ctr.pressed(pynput.keyboard.Key.ctrl, 'w'):
                time.sleep(0.3)
                pass                             
            time.sleep(0.5)            
            pyautogui.doubleClick()  # 双击打开代理人对话窗口
            time.sleep(0.5)

    # 4. 和代理人开始对话
    # 查找[目标完成]
    if find_txt_ocr("目标完成", 3, agent_panel3):
        # 当找到[目标完成],查找[完成任务了]并点击;
        if safe_find_icon("talk2", region=None, max_attempts=3):
            pyautogui.leftClick()
            print("完成任务了...")
            
    time.sleep(0.5)
    print("任务完成,准备返航")
    # 2. 设定返回导航 并出站
    ##navigate_main()
    #print("返航出站...")

if __name__ == "__main__":
    main()