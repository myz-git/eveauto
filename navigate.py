import cv2
import numpy as np
import pyautogui
import pytesseract
import time
import pyperclip  # 导入 pyperclip
from joblib import load
import os
from close import close_icons_main
from outsite import outsite_icons_main

import json


# 在 talk.py 中添加对 close.py 的调用
from outsite import outsite_icons_main

def capture_screen():
    """捕获整个屏幕并转换为OpenCV格式的图像。"""
    screenshot = pyautogui.screenshot()
    screen_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screen_image

def predict_icon_status(image, clf, scaler):
    """Use a machine learning model to determine the status of an icon."""
    # 将图像转换为适合模型输入的格式
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist_scaled = scaler.transform([hist.flatten()])
    prediction = clf.predict(hist_scaled)[0]
    return prediction == 1  # Returns True if the icon is active


def find_and_click_icon(template, width, height, clf, scaler, max_attempts=10,offset_x=0,offset_y=0):
    attempts = 0
    while attempts < max_attempts:
        screen = capture_screen()
        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_screen, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > 0.8:
            icon_image = screen[max_loc[1]:max_loc[1]+height, max_loc[0]:max_loc[0]+width]
            if predict_icon_status(icon_image, clf, scaler):
                x=max_loc[0] + width // 2+offset_x
                y=max_loc[1] + height // 2+offset_y
                pyautogui.moveTo(x, y)
                
                print(f"icon detected and clicked.")
                return True
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts}: Icon not found, retrying...")
        time.sleep(1)
    
    print("Failed to find icon after maximum attempts. Exiting...")
    exit(1)




def find_text_and_click(text, timeout=10,region=None,max_attempts=10):
    """在屏幕上找到指定的文本并点击，打印全局坐标位置。"""
    start_time = time.time()
    if region:
        screenshot = pyautogui.screenshot(region=region)
    else:
        screenshot = pyautogui.screenshot()
    
    attempts = 0
    while attempts < max_attempts:
        if time.time() - start_time > timeout:
            print(f"超时，未找到文本: {text}")
            break
        screen = capture_screen()
        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(screen_gray, lang='chi_sim+eng --psm 6',  output_type=pytesseract.Output.DICT)
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
                print(f"找到{text},鼠标已移动到：({x}, {y})")
                pyautogui.leftClick()               
                pyautogui.moveTo(x, y+112)           
                return
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts}: {text} not found, retrying...")
        time.sleep(1)
        print(f"未找到 {text} . Exiting...")
    exit(1)


def load_location_name(tag):
    """从 JSON 格式的文件中读取位置名称。"""
    try:
        with open('addr.txt', 'r', encoding='utf-8-sig') as file:
            content = file.read()
            print(content)  # 打印文件内容看是否正确读取
            data = json.loads(content)
            return data.get(tag)  # 获取 addr 键对应的值
    except FileNotFoundError:
        print("文件未找到。")
    except json.JSONDecodeError:
        print("解析 JSON 时出错。")
    except UnicodeDecodeError:
        print("文件编码问题，无法读取。")
    return None  # 如果发生错误或找不到 <{tag}>，返回 None




def main():
    # 执行关闭窗口操作
    time.sleep(2)
    close_icons_main()


    # 加载模型和标准化器，确保模型路径正确
    clf_kjz1 = load('model/trained_model_kjz1.joblib')
    scaler_kjz1 = load('model/scaler_kjz1.joblib')
    icon_path_kjz1 = os.path.join('icon', 'kjz1-1.png')
    
    # 加载和处理图标模板
    template_kjz1 = cv2.imread(icon_path_kjz1, cv2.IMREAD_COLOR)
    template_gray_kjz1 = cv2.cvtColor(template_kjz1, cv2.COLOR_BGR2GRAY)
    w_kjz1, h_kjz1 = template_gray_kjz1.shape[::-1]


    clf_search1 = load('model/trained_model_search1.joblib')
    scaler_search1 = load('model/scaler_search1.joblib')
    icon_path_search1 = os.path.join('icon', 'search1-1.png')
  
    template_search1 = cv2.imread(icon_path_search1, cv2.IMREAD_COLOR)
    template_gray_search1 = cv2.cvtColor(template_search1, cv2.COLOR_BGR2GRAY)
    w_search1, h_search1 = template_gray_search1.shape[::-1]
    

    clf_zhongdian1 = load('model/trained_model_zhongdian1.joblib')
    scaler_zhongdian1 = load('model/scaler_zhongdian1.joblib')
    icon_path_zhongdian1 = os.path.join('icon', 'zhongdian1-1.png')
  
    template_zhongdian1 = cv2.imread(icon_path_zhongdian1, cv2.IMREAD_COLOR)
    template_gray_zhongdian1 = cv2.cvtColor(template_zhongdian1, cv2.COLOR_BGR2GRAY)
    w_zhongdian1, h_zhongdian1 = template_gray_zhongdian1.shape[::-1]


    # 1. 按 'l' 键打开地点搜索窗口
    time.sleep(2)  # 等待窗口打开
    pyautogui.press('l')
    time.sleep(1)  # 等待窗口打开

    # 2. 检测并点击 "搜索"
    #find_text_and_click("搜索")
    find_and_click_icon(template_gray_search1, w_search1, h_search1, clf_search1, scaler_search1,10,22)  
    pyautogui.leftClick()

    # 3. 输入地点名称并回车
    location_name = load_location_name('addr')
    if location_name:
        pyperclip.copy(location_name)  # 复制地点名称到剪贴板
        pyautogui.hotkey('ctrl', 'v')  # 粘贴地点名称
        pyautogui.press('enter')
        time.sleep(1)  # 等待搜索结果
    
    # 4. 检测并点击空间站图标
    if find_and_click_icon(template_gray_kjz1, w_kjz1, h_kjz1, clf_kjz1, scaler_kjz1,3,0,22) :
        pyautogui.rightClick()
        time.sleep(0.5) 
        # 5. 检测并点击 "设定为终点"
        find_and_click_icon(template_gray_zhongdian1, w_zhongdian1, h_zhongdian1, clf_zhongdian1, scaler_zhongdian1)
        time.sleep(0.5) 
        pyautogui.leftClick()
        time.sleep(0.5) 
        print("Navigation set successfully.")

    close_icons_main()
    pyautogui.hotkey('ctrl', 'w')
    #出站
    outsite_icons_main()


if __name__ == "__main__":
    main()