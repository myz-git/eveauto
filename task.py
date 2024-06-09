import cv2
import numpy as np
import pyautogui
import pytesseract
import time
from joblib import load
# 在 talk.py 中添加对 close.py 的调用
from close import close_icons_main

import os
import json
from cnocr import CnOcr


def scollscreen(max_attempts=10):
    """转动屏幕"""
    fx,fy=pyautogui.size()
    pyautogui.moveTo(fx/2,fy/2,0.2)
    pyautogui.dragRel(-50,0,0.4,pyautogui.easeOutQuad)    
    print(f"当前位置{pyautogui.position()}划一下...")

def capture_full_screen():
    """捕获屏幕全部区域的截图并转换为OpenCV格式"""
    screenshot = pyautogui.screenshot()
    screen_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screen_image

def capture_screen_area(region):
    """捕获屏幕上指定区域的截图并转换为OpenCV格式"""
    screenshot = pyautogui.screenshot(region=region)
    screen_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screen_image


def predict_icon_status(image, clf, scaler):
    """通过机器学习的模型来验证图标状态"""
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist_scaled = scaler.transform([hist.flatten()])
    prediction = clf.predict(hist_scaled)[0]
    return prediction == 1  # Returns True if the icon is active

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




def find_and_click_icon(template, width, height, clf, scaler, max_attempts=10,offset_x=0,offset_y=0,region=None,):
    """查找图标并移动鼠标至图标上"""  
    attempts = 0
    while attempts < max_attempts:
        scollscreen()  
        screen = capture_screen_area(region)

        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_screen, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > 0.8:
            icon_image = screen[max_loc[1]:max_loc[1]+height, max_loc[0]:max_loc[0]+width]
            if predict_icon_status(icon_image, clf, scaler):
                x=max_loc[0] + width // 2+offset_x
                y=max_loc[1] + height // 2+offset_y
                pyautogui.moveTo(x, y)
                pyautogui.rightClick()
                print(f"icon detected and clicked.")
                return True
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts}: not found, retrying...")
        time.sleep(1)
    
    print(f"代理人 Not found after maximum attempts. Exiting...")
    exit(1)


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 应用自适应阈值
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return processed


def find_agent_and_interact(agent_name,  max_attempts=10, region=None):
    """使用OCR在屏幕特定区域查找代理人名字，并进行交互"""
    attempts = 0
    while attempts < max_attempts:
        scollscreen() 
        
        # 捕获屏幕区域图像
        screen = capture_screen_area(region)
        screen_image = cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2RGB)
        #对当前区域截图并弹出窗口 用于调试
        #cv2.imshow('Captured Area', screen)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        # 初始化OCR工具
        ocr = CnOcr()

        # 执行OCR
        res = ocr.ocr(screen_image)  # 使用 ocr 方法处理整个图像

        # 打印OCR结果
        #print("OCR results:", res)

        # 遍历每一行的识别结果
        for line in res:
            if agent_name in line['text']:
                # 假设我们可以获取到文字的位置
                x = region[0] + line['position'][0][0] + (line['position'][1][0] - line['position'][0][0]) // 2
                y = region[1] + line['position'][0][1] + (line['position'][2][1] - line['position'][0][1]) // 2
            
                # 移动鼠标并点击代理人名字
                pyautogui.moveTo(x, y)
                
                print(f"Interacted with agent {agent_name} at position ({x}, {y}).")
                return True
        #print(data)  # 打印所有识别到的文本，看是否包括目标文本
        time.sleep(1)
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts}: {agent_name} not found, retrying...")
        

    print(f"{agent_name} not found after maximum attempts. Exiting...")
    exit(1)


def main():
    # Load models and scalers
    # 加载模型和标准化器
    clf_agent1 = load('model/trained_model_agent1.joblib')
    scaler_agent1 = load('model/scaler_agent1.joblib')
    
    clf_agent2 = load('model/trained_model_agent2.joblib')
    scaler_agent2 = load('model/scaler_agent2.joblib')

    clf_agent3 = load('model/trained_model_agent3.joblib')
    scaler_agent3 = load('model/scaler_agent3.joblib')

    # 构建图标路径
    icon_path_agent1 = os.path.join('icon', 'agent1-1.png')
    icon_path_agent2 = os.path.join('icon', 'agent2-1.png')
    icon_path_agent3 = os.path.join('icon', 'agent3-1.png')
    
    #
    template_agent1 = cv2.imread(icon_path_agent1, cv2.IMREAD_COLOR)
    template_agent2 = cv2.imread(icon_path_agent2, cv2.IMREAD_COLOR)
    template_agent3 = cv2.imread(icon_path_agent3, cv2.IMREAD_COLOR)

    #
    template_gray_agent1 = cv2.cvtColor(template_agent1, cv2.COLOR_BGR2GRAY)
    template_gray_agent2 = cv2.cvtColor(template_agent2, cv2.COLOR_BGR2GRAY)
    template_gray_agent3 = cv2.cvtColor(template_agent3, cv2.COLOR_BGR2GRAY)

    #
    w_agent1, h_agent1 = template_gray_agent1.shape[::-1]
    w_agent2, h_agent2 = template_gray_agent2.shape[::-1]
    w_agent3, h_agent3 = template_gray_agent3.shape[::-1]

    #设置需要捕获的屏幕区域
    x0, y0, width0, height0 = 1450, 250, 500, 500
    region0=(x0,y0,width0,height0)

    #代理人列表窗口
    x1, y1, width1, height1 = 1650, 450, 200, 50
    region1=(x1,y1,width1,height1)

    # 1. 准备窗口打开
    time.sleep(2)

    # 2. 查找"代理人"图标
    if find_and_click_icon(template_gray_agent1, w_agent1, h_agent1, clf_agent1, scaler_agent1,10,0,0,region0):
        pyautogui.leftClick()
    
    # 3. 查找代理人
    # 3.1 获得代理人名字
    agent_name = load_location_name('agent')
    print(f"agent={agent_name}")
    # 3.2 通过OCR文字识别查找代理人
    if find_agent_and_interact(agent_name,5,region1):
        pyautogui.hotkey('ctrl', 'w')
        pyautogui.doubleClick()  # 双击打开代理人对话窗口

    # 4. 和代理人对话接任务-"我要执行新任务"
    if find_and_click_icon(template_gray_agent2, w_agent2, h_agent2, clf_agent2, scaler_agent2,5):
        pyautogui.leftClick()

    # 5. 和代理人对话接任务-"接受任务"
    if find_and_click_icon(template_gray_agent3, w_agent3, h_agent3, clf_agent3, scaler_agent3,5):
        pyautogui.leftClick()

    print("Program ended.")

if __name__ == "__main__":
    main()
