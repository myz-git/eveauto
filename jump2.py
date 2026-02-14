import numpy as np
import pyautogui
import time
import pynput
import sys
import logging
from say import speak
from utils import log_message, safe_find_icon, hscollscreen, rolljump2, screen_regions, find_txt_ocr

def main():
    """主函数，整合导航和自动驾驶"""
    log_message("INFO", "jump2.py 运行开始", screenshot=False)

    # 屏幕区域配置
    region_full_right = screen_regions['full_right_panel']
    mid_left_panel = screen_regions['mid_left_panel']

    ctr = pynput.keyboard.Controller()
    state = "set_destination"
    find_gate_attempts = 0
    max_find_gate_attempts = 30

    while True:
        if state == "set_destination":
            if safe_find_icon("zhongdian2", mid_left_panel, max_attempts=2):
                log_message("INFO", "终点设置成功，切换到check_local状态", screenshot=False)
                logging.info("终点设置成功，切换到check_local状态")
                state = "check_local"
            else:
                # log_message("ERROR", "未找到终点", screenshot=False)
                state = "find_gate"
                # return 1

        elif state == "check_local":
            if safe_find_icon("tingkao1", mid_left_panel, max_attempts=2):
                log_message("INFO", "找到tingkao1，切换到check_dock状态", screenshot=False)
                logging.info("找到tingkao1，切换到check_dock状态")
                state = "check_dock"
            else:
                log_message("INFO", "未找到tingkao1，切换到find_gate状态", screenshot=True)
                logging.info("未找到tingkao1，切换到find_gate状态")
                state = "find_gate"

        elif state == "find_gate":
            if safe_find_icon("jump6", region_full_right, max_attempts=2) or safe_find_icon("jump5", region_full_right, max_attempts=2) or safe_find_icon("jump4", region_full_right, max_attempts=2) or safe_find_icon("jump0", region_full_right, max_attempts=2):
                log_message("INFO", "找到跳跃门，切换到warp状态", screenshot=False)
                logging.info("找到跳跃门，切换到warp状态")
                state = "warp"
                find_gate_attempts = 0
            else:
                hscollscreen()
                find_gate_attempts += 1
                if find_gate_attempts >= max_find_gate_attempts:
                    log_message("ERROR", f"find_gate尝试{max_find_gate_attempts}次失败", screenshot=True)
                    logging.error(f"find_gate尝试{max_find_gate_attempts}次失败")
                    return 1
                log_message("INFO", f"find_gate尝试第{find_gate_attempts}次", screenshot=False)

        elif state == "warp":
            # 先找jump3
            if safe_find_icon("jump3", region_full_right, max_attempts=1):
                log_message("INFO", "找到jump3，切换到check_dock状态", screenshot=False)
                state = "check_dock"
            else:
                # 找不到jump3，使用rolljump2
                result = rolljump2()
                if result == 0:  # 到达目的地
                    log_message("INFO", "已到达目的地,程序停止", screenshot=True)
                    logging.info("已到达目的地,程序停止")
                    return 0
                elif result:  # 找到jump3
                    log_message("INFO", "rolljump2成功，切换到check_dock状态", screenshot=False)
                    logging.info("rolljump2成功，切换到check_dock状态")
                    state = "check_dock"
                else:
                    log_message("INFO", "rolljump2继续尝试", screenshot=False)

        elif state == "check_dock":
            # if safe_find_icon("out1", region_full_right, max_attempts=30,threshold=0.7, cnn_threshold=0.60, action=None):
            while True:
                safe_find_icon("jump3", region_full_right, max_attempts=1)
                # speak("已切换到check_dock状态,等待完成停靠")
                if find_txt_ocr("离站", max_attempts=1, region=region_full_right):
                    log_message("INFO", "空间站已停靠，jump2运行结束", screenshot=False)
                    logging.info("空间站已停靠，jump2运行结束")
                    time.sleep(2)
                    return 0

        time.sleep(1)

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        log_message("ERROR", f"全局异常: {e}", screenshot=True)
        sys.exit(1)