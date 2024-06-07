import cv2
import numpy as np
import pyautogui
import pytesseract
import time
import pyperclip  # 导入 pyperclip
from joblib import load
import os

def capture_screen():
    """捕获整个屏幕并转换为OpenCV格式的图像。"""
    screenshot = pyautogui.screenshot()
    screen_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screen_image

def find_text_and_click(text, timeout=3,region=None):
    while True:
        # 截取屏幕指定区域
        screenshot=capture_screen()
        #if region:
        #    screenshot = pyautogui.screenshot(region=region)
        #else:
        #    screenshot = pyautogui.screenshot()

        # 将截图转换为图像数据，以便更详细地处理识别结果
        data = pytesseract.image_to_data(screenshot, lang='chi_sim+eng', config = '--psm 1 --oem 1',output_type=pytesseract.Output.DICT)
        print (data)
        # 检查是否找到特定的文字
        for i in range(len(data['text'])):
            if text in data['text'][i]:
                # 计算文字的中心位置
                x = data['left'][i] + data['width'][i] // 2
                y = data['top'][i] + data['height'][i] // 2
                # 如果使用了区域截图，确保加上区域的原点坐标
                if region:
                    x += region[0]
                    y += region[1]
                # 移动鼠标到文字位置
                pyautogui.moveTo(x, y)
                print(f"{text}位置：({x}, {y})")
                pyautogui.leftClick()
                return
        print(f"未找到{text}，继续扫描")
        time.sleep(1)

def find_text_and_click3(text, timeout=3,region=None):
    while True:
        
        screenshot=capture_screen()

        # 将截图转换为图像数据，以便更详细地处理识别结果
        data = pytesseract.image_to_data(screenshot, lang='chi_sim', config='--psm 6', output_type=pytesseract.Output.DICT)

        # 检查是否找到特定的文字
        for i in range(len(data['text'])):
            if text in data['text'][i]:
                # 计算文字的中心位置
                x = data['left'][i] + data['width'][i] // 2
                y = data['top'][i] + data['height'][i] // 2
                # 如果使用了区域截图，确保加上区域的原点坐标
                if region:
                    x += region[0]
                    y += region[1]
                # 移动鼠标到文字位置
                pyautogui.moveTo(x, y)
                print(f"{text}位置：({x}, {y})")
                pyautogui.rightClick()
                return
        print(f"未找到{text}，继续扫描")
        time.sleep(1)

def find_text_and_click2(text, timeout=3):
    """在屏幕上找到指定的文本的第二个匹配项并右键点击，打印全局坐标位置。"""
    start_time = time.time()
    found_count = 0  # 用来记录找到文本的次数
    while True:
        if time.time() - start_time > timeout:
            print(f"超时，未找到文本: {text}")
            break
        screen = capture_screen()
        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(screen_gray, lang='chi_sim', config='--psm 6', output_type=pytesseract.Output.DICT)
        for i in range(len(data['text'])):
            if text in data['text'][i]:
                found_count += 1
                if found_count == 2:  # 当找到第二次匹配时执行点击
                    # 计算文字的中心位置
                    x = data['left'][i] + data['width'][i] // 2
                    y = data['top'][i] + data['height'][i] // 2
                    # 移动鼠标到文字位置并右键点击
                    pyautogui.moveTo(x, y)
                    
                    print(f"右键点击了第二个匹配的 '{text}' 位置：({x}, {y})")
                    return True
        print("未找到文字，继续扫描")
        time.sleep(1)
    return False

def load_location_name():
    """从文件读取位置名称。尝试不同的编码方式打开文件。"""
    try:
        with open('addr.txt', 'r', encoding='utf-8') as file:
            line = file.readline()
            prefix = 'addr='
            if line.startswith(prefix):
                return line[len(prefix):].strip('" \n')
    except UnicodeDecodeError:
        # 如果 UTF-8 失败，尝试使用 GBK 编码读取
        with open('addr.txt', 'r', encoding='gbk') as file:
            line = file.readline()
            prefix = 'addr='
            if line.startswith(prefix):
                return line[len(prefix):].strip('" \n')


def main():
    # 1. 按 'l' 键打开地点搜索窗口
    time.sleep(3)  # 等待窗口打开
    pyautogui.press('l')
    time.sleep(1)  # 等待窗口打开

    # 2. 检测并点击 "搜索"
    print("查找搜素...")    
    find_text_and_click("搜索")


    # 3. 输入地点名称并回车
    
    location_name = load_location_name()
    if location_name:
        pyperclip.copy(location_name)  # 复制地点名称到剪贴板
        pyautogui.hotkey('ctrl', 'v')  # 粘贴地点名称
        pyautogui.press('enter')
        time.sleep(1)  # 等待搜索结果
    
    # 4. 在搜索结果中找到地点并右键点击第二个匹配项
    location_name = "空间"
    if find_text_and_click(location_name):
        pyautogui.rightClick()
        print("第二个地点位置已经被成功右键点击。")
    else:
        print("未找到第二个匹配的地点位置。")

    # 5. 检测并点击 "设定为终点"
    find_text_and_click("设定为终点")

    print("Navigation set successfully.")

if __name__ == "__main__":
    main()
