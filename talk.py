import pyautogui
import time
import pynput
import logging
import sys

# 内部程序调用
from say import speak
from utils import scollscreen, capture_screen_area, safe_find_icon, find_txt_ocr, find_txt_ocr2, screen_regions, close_icons_main, log_message
from dbcfg import get_location_name  # 修改：导入 get_location_name
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
    log_message("INFO", "talk.py 开始运行", screenshot=False)
    
    # 查找[开始对话]
    if safe_find_icon("talk1", mid_left_panel, max_attempts=3):
        pyautogui.leftClick()        
        print("开始对话...")
        log_message("INFO", "找到[talk1]图标并点击，开始对话", screenshot=False)
        time.sleep(1)
    else:
        # 2. 查找"代理人"图标  
        if safe_find_icon("agent1", agent_panel1, max_attempts=2):
            pyautogui.leftClick()
            log_message("INFO", "找到[agent1]图标并点击", screenshot=False)
    
        # 3. 查找代理人
        # 3.1 获得代理人名字
        try:
            agent_name = get_location_name('agent')
            print(f"agent={agent_name}")
            log_message("INFO", f"代理人名称: {agent_name}", screenshot=False)
        except ValueError as e:
            log_message("ERROR", f"加载代理人名称失败: {e}", screenshot=True)
            sys.exit(1)

        # 3.2 通过OCR文字识别查找代理人
        if find_txt_ocr(agent_name, 1, agent_panel2):
            with ctr.pressed(pynput.keyboard.Key.ctrl, 'w'):
                time.sleep(0.3)
                pass                             
            time.sleep(0.5)            
            pyautogui.doubleClick()
            time.sleep(0.5)
            log_message("INFO", f"找到代理人 {agent_name}，双击打开对话窗口", screenshot=False)
        else:
            log_message("ERROR", f"未找到代理人 {agent_name}", screenshot=True)
            sys.exit(1)

    # 4. 和代理人开始对话
    # 查找[目标完成]
    if find_txt_ocr("目标完成", 3, agent_panel3):
        # 当找到[目标完成],查找[完成任务了]并点击
        if safe_find_icon("talk2", region=None, max_attempts=3):
            pyautogui.leftClick()
            print("完成任务了...")
            log_message("INFO", "找到[talk2]图标并点击，任务完成", screenshot=False)
        else:
            log_message("ERROR", "未找到[talk2]图标", screenshot=True)
            sys.exit(1)
    else:
        log_message("ERROR", "未找到[目标完成]文本", screenshot=True)
        sys.exit(1)

    time.sleep(0.5)
    print("任务完成,准备返航")
    log_message("INFO", "任务完成，准备返航", screenshot=False)
    # 设定返回导航并出站
    # navigate_main()
    # print("返航出站...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_message("ERROR", f"talk.py 全局异常: {e}", screenshot=True)
        sys.exit(1)