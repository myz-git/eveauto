import cv2
import numpy as np
import pyautogui
import pytesseract
import time
from joblib import load
# 在 talk.py 中添加对 close.py 的调用
from close import close_icons_main
from cnocr import CnOcr

import os
def scollscreen(max_attempts=10):
    """转动屏幕"""
    fx,fy=pyautogui.size()
    pyautogui.moveTo(fx/2,fy/2+270,0.2)
    pyautogui.dragRel(-50,0,0.4,pyautogui.easeOutQuad)    

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
    """Use a machine learning model to determine the status of an icon."""
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist_scaled = scaler.transform([hist.flatten()])
    prediction = clf.predict(hist_scaled)[0]
    return prediction == 1  # Returns True if the icon is active


def find_and_click_icon(template, width, height, clf, scaler, max_attempts=10,offset_x=0,offset_y=0,region=None,):
    """查找图标并移动鼠标至图标上"""  
    
    # 如果没有提供 region，使用全屏作为默认区域
    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)

    attempts = 0
    while attempts < max_attempts:
        scollscreen()  
        screen = capture_screen_area(region)

        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_screen, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        #对当前区域截图并弹出窗口 用于调试
        """
        cv2.imshow('Captured Area', screen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

        if max_val > 0.8:
            icon_image = screen[max_loc[1]:max_loc[1]+height, max_loc[0]:max_loc[0]+width]
            if predict_icon_status(icon_image, clf, scaler):
                x=max_loc[0] + region[0] + width // 2 + offset_x
                y=max_loc[1] + region[1] + height // 2 + offset_y
                pyautogui.moveTo(x, y)
                """
                print(f"max_loc[0]:{max_loc[0]},width:{width}") 
                print(f"max_loc[1]:{max_loc[1]},height:{height}") 
                print(f"moveTo:{x,y}")
                """
                print("Icon detected!")
                return True
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts}: not found, Retrying...")
        time.sleep(1)
    
    print("Icon Not found after maximum attempts. 程序终止...")
    # 程序退出
    exit(1)


def find_txt_ocr(txt,  max_attempts=10, region=None):
    """使用OCR在屏幕特定区域查找txt内容"""
    attempts = 0
    while attempts < max_attempts:
        
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
            if txt in line['text']:
                # 假设我们可以获取到文字的位置
                x = region[0] + line['position'][0][0] + (line['position'][1][0] - line['position'][0][0]) // 2
                y = region[1] + line['position'][0][1] + (line['position'][2][1] - line['position'][0][1]) // 2
            
                # 移动鼠标并点击代理人名字
                pyautogui.moveTo(x, y)
                
                print(f"{txt} dected at position ({x}, {y}).")
                return True
        #print(data)  # 打印所有识别到的文本，看是否包括目标文本
        time.sleep(1)
        scollscreen() 
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts}: {txt} not found, Retrying...")
        

    print(f"{txt} not found after maximum attempts. 程序终止!")
    exit(1)


def lock_and_fight(txt,  max_attempts=10, region=None):
    """使用OCR在屏幕特定区域查找txt内容"""
    attempts = 0
    while attempts < max_attempts:
        
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
            if txt in line['text']:
                # 假设我们可以获取到文字的位置
                x = region[0] + line['position'][0][0] + (line['position'][1][0] - line['position'][0][0]) // 2
                y = region[1] + line['position'][0][1] + (line['position'][2][1] - line['position'][0][1]) // 2
            
                # 移动鼠标并点击代理人名字
                pyautogui.moveTo(x, y)
                
                print(f"{txt} dected at position ({x}, {y}).")
                return True
        #print(data)  # 打印所有识别到的文本，看是否包括目标文本
        time.sleep(1)
        scollscreen() 
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts}: {txt} not found, Retrying...")
        

    print(f"{txt} not found after maximum attempts. 程序终止!")
    exit(1)


def main():
    # Load models and scalers
    clf_talk1 = load('model/trained_model_talk1.joblib')
    scaler_talk1 = load('model/scaler_talk1.joblib')
    clf_talk2 = load('model/trained_model_talk2.joblib')
    scaler_talk2 = load('model/scaler_talk2.joblib')
    clf_out1 = load('model/trained_model_out1.joblib')
    scaler_out1 = load('model/scaler_out1.joblib')

    # Define paths to icon templates
    icon_path_talk1 = os.path.join('icon', 'talk1-1.png')
    icon_path_talk2 = os.path.join('icon', 'talk2-1.png')
    icon_path_out1 = os.path.join('icon', 'out1-1.png')

    # Load and process icon templates
    template_talk1 = cv2.imread(icon_path_talk1, cv2.IMREAD_COLOR)
    template_gray_talk1 = cv2.cvtColor(template_talk1, cv2.COLOR_BGR2GRAY)
    w_talk1, h_talk1 = template_gray_talk1.shape[::-1]
    template_talk2 = cv2.imread(icon_path_talk2, cv2.IMREAD_COLOR)
    template_gray_talk2 = cv2.cvtColor(template_talk2, cv2.COLOR_BGR2GRAY)
    w_talk2, h_talk2 = template_gray_talk2.shape[::-1]
    template_out1 = cv2.imread(icon_path_out1, cv2.IMREAD_COLOR)
    template_gray_out1 = cv2.cvtColor(template_out1, cv2.COLOR_BGR2GRAY)
    w_out1, h_out1 = template_gray_out1.shape[::-1]


    # 设置需要捕获的屏幕区域
    fx,fy=pyautogui.size()
    #右侧面板All
    x0, y0, width0, height0 = 1300, 50, 600, 1000
    region0=(x0,y0,width0,height0)

    #右侧面板2(不含右上角选择物体)
    x1, y1, width1, height1 = 1300, 150, 600, 1000
    region1=(x1,y1,width1,height1)
    

    
    # 1. 准备开始
    time.sleep(2)
    scollscreen()

    # 查找[SHUIPS]
    if find_txt_ocr("SHUIPS",5,region1):
        # 当找到[通用],并点击;
        pyautogui.leftClick()
        print("找到[SHUIPS]了...")
        time.sleep(1)

    # 查找[小行星(水硼]
    if find_txt_ocr("小行星(水硼",5,region1):
        # 当找到[通用],并点击;
        pyautogui.leftClick()
        time.sleep(0.1)
        print("找到[小行星(水硼]了...")

        #环绕目标
        pyautogui.hotkey('w')
        time.sleep(0.1)

        # 锁定目标
        #pyautogui.hotkey('ctrl')
        #time.sleep(1)
        pyautogui.rightClick()
        time.sleep(0.1)
        if lock_and_fight("锁定目标",3,region1):
            pyautogui.leftClick()
            time.sleep(0.2)
            pyautogui.hotkey('F1')
            pyautogui.hotkey('F2')
            
            while True:
                time.sleep(30)
                pyautogui.hotkey('F1')
                pyautogui.hotkey('F2')
                time.sleep(3)
                pyautogui.hotkey('F1')
                pyautogui.hotkey('F2')


        
    print("任务结束...")
    
if __name__ == "__main__":
    main()
