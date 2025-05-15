import numpy as np
import pyautogui
import pyperclip
import time
from joblib import load
from cnocr import CnOcr
import re
import pynput
from utils import scollscreen, capture_screen_area, safe_find_icon, load_location_name, find_txt_ocr, find_txt_ocr2, correct_string, screen_regions, close_icons_main, log_message  # 新增：导入log_message

# 内部程序调用
from say import speak
from outsite import outsite_icons_main, outsite_check

class IconNotFoundException(Exception):
    """Exception raised when an icon is not found."""
    pass

class GoodsNotFoundException(Exception):
    """Exception raised when the specified goods are not found."""
    pass

class TextNotFoundException(Exception):
    """Exception raised when the specified text is not found."""
    pass

def extract_goods_name(text):
    """从文本中提取指定格式的商品名称"""
    pattern = re.compile(r'货物\d*×(.+?)\*')
    match = pattern.search(text)
    if match:
        goods_name = match.group(1).strip()
        log_message("INFO", f"提取货物名称: {goods_name}")  # 新增：记录提取结果
        return goods_name
    log_message("DEBUG", "未提取到货物名称")  # 新增：记录未提取
    return None

def get_goods(icon, max_attempts=3, offset_x=0, offset_y=0, region=None):
    """定位运输目标"""
    if safe_find_icon(icon, region, max_attempts=max_attempts, offset_x=offset_x, offset_y=offset_y,threshold=0.7,cnn_threshold=0.7):
        print("--查找运输目标")
        log_message("INFO", f"查找运输目标图标: {icon}", screenshot=False)  # 新增：记录查找
        time.sleep(0.2)
        # 获取货物内容, 根据运输目标坐标直接定位货物右侧的OCR扫描范围
        x, y = pyautogui.position()
        region_hw = (x + 80, y + 94, 280, 50)
        print(f"--region_hw={region_hw}")
        log_message("DEBUG", f"OCR扫描区域: {region_hw}")  # 新增：记录区域
        
        # 只匹配汉字，如果折行只提取第一行
        goods = find_txt_ocr2('', 3, region_hw)
        log_message("INFO", f"OCR识别货物: {goods}", screenshot=False)  # 新增：记录结果
        return goods
    log_message("ERROR", f"未找到运输目标图标: {icon}", screenshot=True)  # 新增：记录失败
    raise GoodsNotFoundException(f"未找到运输目标: {icon}")

def move_goods(goods):
    # 从仓库搬运货物到舰船机库
    # 打开仓库
    print("3.1 打开仓库")
    log_message("INFO", "打开仓库")  # 新增：记录操作
    ctr = pynput.keyboard.Controller()
    with ctr.pressed(pynput.keyboard.Key.alt, 'c'):
        time.sleep(0.2)
        pass
    log_message("DEBUG", "按下Alt+C打开仓库")  # 新增：记录键盘操作
    time.sleep(0.2)
    cangku_panel3 = screen_regions['cangku_panel3']

    # 获取机库坐标
    print("3.2 找机库坐标")
    if safe_find_icon("jiku1", cangku_panel3, max_attempts=3, offset_x=20, offset_y=0):
        print("3.2.1 找机库坐标")
        jiku_x, jiku_y = pyautogui.position()
        log_message("INFO", f"找到机库图标[jiku1]，坐标: ({jiku_x}, {jiku_y})", screenshot=False)  # 新增：记录坐标
    else:
        log_message("ERROR", "未找到机库图标[jiku1]", screenshot=True)  # 新增：记录失败
        return

    # 激活仓库
    if safe_find_icon("jiku2", cangku_panel3, max_attempts=2):
        print("3.2.2 激活仓库")
        pyautogui.leftClick()
        time.sleep(0.2)
        log_message("INFO", "找到[jiku2]图标并点击，激活仓库", screenshot=False)  # 新增：记录点击
    else:
        log_message("ERROR", "未找到[jiku2]图标", screenshot=True)  # 新增：记录失败
        return

    # 搜索仓库
    print("3.3 查找仓库搜索栏")
    if safe_find_icon("search2", cangku_panel3, max_attempts=5):
        print("3.4 搜索仓库..")
        pyautogui.leftClick()
        log_message("INFO", "找到[search2]图标并点击", screenshot=False)  # 新增：记录点击
        pyperclip.copy(goods[:4])
        pyautogui.leftClick()
        time.sleep(0.2)
        with ctr.pressed(pynput.keyboard.Key.ctrl, 'v'):
            time.sleep(0.2)
            pass
        time.sleep(0.2)
        with ctr.pressed(pynput.keyboard.Key.enter):
            time.sleep(0.2)
            pass
        log_message("DEBUG", f"搜索货物: {goods[:4]}")  # 新增：记录搜索
        time.sleep(1)
        pyautogui.moveRel(-50, 65)
        time.sleep(0.2)
        pyautogui.dragTo(jiku_x + 10, jiku_y, 1, pyautogui.easeOutQuad)
        time.sleep(0.2)
        with ctr.pressed(pynput.keyboard.Key.ctrl, 'w'):
            time.sleep(0.3)
            pass
        time.sleep(0.2)
        print(f"3.5 [{goods}]已放入机库中,请检查...")
        log_message("INFO", f"货物[{goods}]已放入机库", screenshot=False)  # 新增：记录完成
    else:
        log_message("ERROR", "未找到[search2]图标", screenshot=True)  # 新增：记录失败
        return

def check_goods(cangku_panel3):
    print("通过OCR文字识别是否有[货柜缺失]弹窗")
    log_message("INFO", "检查是否有[货柜缺失]弹窗")  # 新增：记录检查
    try:
        # 尝试找到 "货柜缺失" 文本
        ctr = pynput.keyboard.Controller()
        if find_txt_ocr('货柜缺失', 1, cangku_panel3):
            pyautogui.leftClick()
            time.sleep(0.2)
            with ctr.pressed(pynput.keyboard.Key.ctrl, 'w'):
                time.sleep(0.3)
                pass
            time.sleep(0.2)
            log_message("INFO", "找到[货柜缺失]弹窗并关闭", screenshot=True)  # 新增：记录关闭
            return False
        log_message("DEBUG", "未找到[货柜缺失]弹窗，货物正常")  # 新增：记录正常
        return True
    except TextNotFoundException:
        log_message("DEBUG", "未找到[货柜缺失]文本，货物正常")  # 新增：记录正常
        return True

def main():
    # 加载键盘控制器
    ctr = pynput.keyboard.Controller()
    log_message("INFO", "task.py 开始运行", screenshot=False)  # 新增：记录开始

    # 设置需要捕获的屏幕区域
    region_full_right = screen_regions['full_right_panel']
    agent_panel1 = screen_regions['agent_panel1']
    agent_panel2 = screen_regions['agent_panel2']
    agent_panel3 = screen_regions['agent_panel3']
    need_goods_panel = screen_regions['need_goods_panel']
    cangku_panel3 = screen_regions['cangku_panel3']

    # 01. 准备开始
    print("task start")
    time.sleep(1)
    print("0 task sleep(1)")
    log_message("INFO", "task.py 准备开始，等待1秒")  # 新增：记录等待

    # 02. 查找"代理人"图标
    print("0.2.0 查找[代理人]图标")
    if safe_find_icon("agent1", agent_panel1, max_attempts=2):
        pyautogui.leftClick()
        log_message("INFO", "找到[代理人]图标并点击", screenshot=False)  # 新增：记录点击
    else:
        log_message("ERROR", "未找到[代理人]图标", screenshot=True)  # 新增：记录失败

    # 03. 查找代理人
    # 0.3.1 获得代理人名字
    print("0.3.0 获得代理人名字")
    agent_name = load_location_name('agent')
    print(f"0.3.1 agent={agent_name}")
    log_message("INFO", f"加载代理人名称: {agent_name}")  # 新增：记录代理人

    # 0.3.2 通过OCR文字识别查找代理人
    print(f"0.3.2 通过OCR文字识别查找代理人")
    if find_txt_ocr(agent_name, 1, agent_panel2):
        with ctr.pressed(pynput.keyboard.Key.ctrl, 'w'):
            time.sleep(0.3)
            pass
        time.sleep(0.2)
        pyautogui.doubleClick()
        time.sleep(0.2)
        log_message("INFO", f"找到代理人: {agent_name}，双击打开对话窗口", screenshot=False)  # 新增：记录操作
    else:
        log_message("ERROR", f"未找到代理人: {agent_name}", screenshot=True)  # 新增：记录失败

    # 0.4. 和代理人开始对话
    """↓↓↓ 低安特供 ↓↓↓"""
    try:
        # goods = get_goods("yunshumubiao1", max_attempts=2, region=agent_panel3)
        goods = get_goods("yunshumubiao1", max_attempts=2)
        print(f"0.5.0 低安任务直接获得货物:{goods}")
        log_message("INFO", f"低安任务获得货物: {goods}")  # 新增：记录货物
        # 查找[接受]任务按钮
        if safe_find_icon("agent3", region=None,max_attempts=3):
            pyautogui.leftClick()
            print("0.5.1 低安已接受任务!")
            log_message("INFO", "低安任务接受成功", screenshot=False)  # 新增：记录接受
            with ctr.pressed(pynput.keyboard.Key.ctrl, 'w'):
                time.sleep(0.3)
                pass
            time.sleep(0.2)
            log_message("DEBUG", "关闭窗口（Ctrl+W）")  # 新增：记录关闭
    except GoodsNotFoundException as e:
        print(e)
        log_message("ERROR", f"低安任务货物识别失败: {e}", screenshot=True)  # 新增：记录错误

    # 当还未提供任务
    print("1.0 正式开始读取任务")
    if safe_find_icon("agent2", region_full_right,max_attempts=2):
        print("1.1 我要执行新任务")
        pyautogui.leftClick()
        time.sleep(0.2)
        log_message("INFO", "找到[agent2-1]图标并点击，开始新任务", screenshot=False)  # 新增：记录点击
        #...get goods
        try:
            goods = get_goods("yunshumubiao1", max_attempts=2, region=agent_panel3)
            print(f"1.2 获得货物:{goods}")
            log_message("INFO", f"获得货物: {goods}")  # 新增：记录货物
        except GoodsNotFoundException as e:
            print(e)
            log_message("ERROR", f"货物识别失败: {e}", screenshot=True)  # 新增：记录错误
        
        # 查找[接受]任务按钮
        if safe_find_icon("agent3", region_full_right,max_attempts=2):
            pyautogui.leftClick()
            print("1.3 已接受任务!")
            log_message("INFO", "任务接受成功", screenshot=False)  # 新增：记录接受
            with ctr.pressed(pynput.keyboard.Key.ctrl, 'w'):
                time.sleep(0.3)
                pass
            time.sleep(0.2)
            log_message("DEBUG", "关闭窗口（Ctrl+W）")  # 新增：记录关闭

    # 查看任务
    else:
        # 当有查看任务时
        if safe_find_icon("chakanrenwu1", region_full_right,max_attempts=2):
            print("2.1 查看任务")
            pyautogui.leftClick()
            time.sleep(0.2)
            log_message("INFO", "找到[查看任务]图标并点击", screenshot=False)  # 新增：记录点击
            try:
                goods = get_goods("yunshumubiao1", max_attempts=3, region=agent_panel3)
                print(f"2.2 获得货物:{goods}")
                log_message("INFO", f"获得货物: {goods}")  # 新增：记录货物
            except GoodsNotFoundException as e:
                print(e)
                log_message("ERROR", f"货物识别失败: {e}", screenshot=True)  # 新增：记录错误
            
            # 查找[接受]任务按钮
            if safe_find_icon("agent3", region_full_right,max_attempts=2):
                pyautogui.leftClick()
                print("2.3 已接受任务!")
                log_message("INFO", "任务接受成功", screenshot=False)  # 新增：记录接受
                with ctr.pressed(pynput.keyboard.Key.ctrl, 'w'):
                    time.sleep(0.3)
                    pass
                time.sleep(0.2)
                log_message("DEBUG", "关闭窗口（Ctrl+W）")  # 新增：记录关闭
            else:
                print("2.4 之前接受过任务!")
                log_message("INFO", "任务已接受，无需再次接受")  # 新增：记录状态
                with ctr.pressed(pynput.keyboard.Key.ctrl, 'w'):
                    time.sleep(0.3)
                    pass
                time.sleep(0.2)
                log_message("DEBUG", "关闭窗口（Ctrl+W）")  # 新增：记录关闭

    # 对OCR识别结果进行修正
    goods = correct_string(goods)
    print(f"任务目标:运送[{goods}]")
    log_message("INFO", f"任务目标: 运送[{goods}]")  # 新增：记录目标

    # 从仓库移动货物到船仓
    time.sleep(0.2)
    print(f"准备搬运:{goods}")
    log_message("INFO", f"准备搬运货物: {goods}")  # 新增：记录搬运
    time.sleep(0.2)
    move_goods(goods)
                                                                   
    # 9. 出站
    close_icons_main()
    time.sleep(1)
    log_message("INFO", "关闭所有窗口，准备出站", screenshot=False)  # 新增：记录出站
    
    print("出站中,请等待...")
    outsite_icons_main()
    outsite_check()
    log_message("INFO", "出站完成", screenshot=False)  # 新增：记录完成

if __name__ == "__main__":
    main()