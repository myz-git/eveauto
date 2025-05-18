import pyautogui
import pyperclip
import time
import pynput
import logging
import sys

# 内部程序调用
from say import speak
from outsite import outsite_icons_main
from utils import safe_find_icon, find_txt_ocr, screen_regions, close_icons_main, log_message
from dbcfg import get_location_name  # 修改：导入 get_location_name

def navigate_main():
    # 执行关闭窗口操作
    time.sleep(1)
    # 中间对话框
    agent_panel3 = screen_regions['agent_panel3']
    upper_left_panel = screen_regions['upper_left_panel']

    log_message("INFO", "navigate.py 开始运行", screenshot=False)

    # 1. 按 'l' 键打开地点搜索窗口   
    try:
        location_name = get_location_name('addr')
        log_message("INFO", f"加载空间站地址: {location_name}", screenshot=False)
    except ValueError as e:
        log_message("ERROR", f"加载空间站地址失败: {e}", screenshot=True)
        sys.exit(1)

    ctr = pynput.keyboard.Controller()
    time.sleep(1)
    # 2. 检测并点击 "搜索"
    if find_txt_ocr('搜索任意内容', region=upper_left_panel):
        pyautogui.leftClick()
        log_message("INFO", "找到[搜索任意内容]文本并点击", screenshot=False)

        # 3. 输入地点名称并回车
        pyperclip.copy(location_name)
        pyautogui.leftClick() 
        time.sleep(1)
        with ctr.pressed(pynput.keyboard.Key.ctrl, 'v'):
            time.sleep(0.8)
            pass 

        time.sleep(1)
        with ctr.pressed(pynput.keyboard.Key.enter):
            time.sleep(0.8)
            pass             

        time.sleep(1)
    
        # 4. 检测并点击空间站图标（全屏区域）
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)
        if safe_find_icon("kjz1", region, max_attempts=2, offset_x=0, offset_y=22):
            pyautogui.rightClick()
            time.sleep(0.5) 
            # 5. 检测并点击 "设定为终点"
            if safe_find_icon("zhongdian1", region, max_attempts=2):
                time.sleep(0.8) 
                pyautogui.leftClick()
                time.sleep(0.8) 
                print("导航已设定!")
                log_message("INFO", "导航设定成功", screenshot=False)
            else:
                log_message("ERROR", "未找到[zhongdian1]图标", screenshot=True)
                sys.exit(1)
        else:
            log_message("ERROR", "未找到[kjz1]图标", screenshot=True)
            sys.exit(1)
    else:
        log_message("ERROR", "未找到[搜索任意内容]文本", screenshot=True)
        sys.exit(1)

    close_icons_main()
    # 关闭窗口(ctrl+w)
    with ctr.pressed(pynput.keyboard.Key.ctrl, 'w'):
        time.sleep(0.3)
        pass

    # 关闭导航窗口
    if find_txt_ocr("关闭", max_attempts=1, region=agent_panel3):
        time.sleep(0.2) 
        pyautogui.leftClick()
        log_message("INFO", "找到[关闭]文本并点击", screenshot=False)

    # 出站
    time.sleep(2) 
    outsite_icons_main()
    log_message("INFO", "出站完成", screenshot=False)

if __name__ == "__main__":
    try:
        navigate_main()
    except Exception as e:
        log_message("ERROR", f"navigate.py 全局异常: {e}", screenshot=True)
        sys.exit(1)