
from cnocr import CnOcr

import cv2
import numpy as np
import pyautogui
import pytesseract
import pyperclip  # 导入 pyperclip
import time
from joblib import load
import os
import json
from cnocr import CnOcr
import re
from model_config import models, templates, screen_regions

class IconNotFoundException(Exception):
    """Exception raised when an icon is not found."""
    pass

class TextNotFoundException(Exception):
    """Exception raised when the specified text is not found."""
    pass

class GoodsNotFoundException(Exception):
    """Exception raised when the specified goods are not found."""
    pass

def capture_screen_area(region, save_path=None):
    """捕获屏幕上指定区域的截图并转换为OpenCV格式，可选地保存到文件"""
    screenshot = pyautogui.screenshot(region=region)
    if save_path:
        screenshot.save(save_path)  # 保存截图到指定路径
    screen_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screen_image

def find_txt_ocr3(txt, region=None, save_path=None):
     if region is None:
          fx, fy = pyautogui.size()
          region = (0, 0, fx, fy)

     ocr = CnOcr(rec_model_name='scene-densenet_lite_246-gru_base')
     #img_fp = 'pic//tmp3.png'
     #ocr = CnOcr()    
     #res = ocr.ocr(img_fp)

     screen_image = capture_screen_area(region, save_path=f"debug_image_.png")  # Save each attempt

     res = ocr.ocr(screen_image)  # 使用 ocr 方法处理整个图像
     #print("OCR results:", res)

     # 遍历每一行的识别结果
     for line in res:
         if txt in line['text']:
             x = region[0] + line['position'][0][0] + (line['position'][1][0] - line['position'][0][0]) // 2
             y = region[1] + line['position'][0][1] + (line['position'][2][1] - line['position'][0][1]) // 2
             print(f"Interacted with text {txt} at position ({x}, {y}).")
             return line['text']

     print(f" {txt} not found, retrying...")

def find_txt_ocr2(txt,  max_attempts=5, region=None):
    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)

    """使用CNOCR高识别率模型,在屏幕特定区域查找"""
    attempts = 0
    while attempts < max_attempts:                
        # 初始化OCR工具
        #ocr = CnOcr()
        ocr = CnOcr(rec_model_name='scene-densenet_lite_246-gru_base')
        
        # 执行OCR
        #screen_image=pyautogui.screenshot(region=region)
        screen_image = capture_screen_area(region)
        #screen_image.save()
        #screen_image=preprocess_image(screen_image)
        res = ocr.ocr(screen_image)  # 使用 ocr 方法处理整个图像

        # 打印OCR结果
        #print("OCR results:", res)

        # 遍历每一行的识别结果
        extracted_goods_name=None

        pattern = re.compile(f'{txt}([^()]+)')
        # 遍历所有字典
        for item in res:
            match = pattern.search(item['text'])
            if match:
                # 找到固定部分的起始位置
                extracted_goods_name = re.sub(r'[^\u4e00-\u9fff]', '', match.group(1))
                print(f"识别文字: {extracted_goods_name}")
                return extracted_goods_name
        time.sleep(1)
        attempts += 1
        #scollscreen()
        print(f"Attempt {attempts}/{max_attempts}: {txt} not found, retrying...")
        
    raise TextNotFoundException(f"{txt} not found after maximum attempts.")


def get_goods(region=None):
     goods=None
     #goods = find_txt_ocr2('种子',3,region)
     goods = find_txt_ocr3('种子', region)
     return goods

def find_icon(template, width, height, clf, scaler, max_attempts=10, offset_x=0, offset_y=0, region=None, exflg=False):
    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)

    attempts = 0
    while attempts < max_attempts:
        
        screen = capture_screen_area(region)
        # 确保模板和搜索图像都是彩色或都是灰度
        # 如果你的模板是彩色的，确保不要转换成灰度
        # 如果你的模板是灰度的，确保图像也是灰度
        # res = cv2.matchTemplate(cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY), template, cv2.TM_CCOEFF_NORMED)
        res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)  # 使用彩色图像进行模板匹配
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print(f"Attempt {attempts + 1}, max_val: {max_val}")  # 打印当前的最大匹配值

        if max_val > 0.8:
            icon_image = screen[max_loc[1]:max_loc[1] + height, max_loc[0]:max_loc[0] + width]
            if predict_icon_status(icon_image, clf, scaler):
                x = max_loc[0] + region[0] + width // 2 + offset_x
                y = max_loc[1] + region[1] + height // 2 + offset_y
                pyautogui.moveTo(x, y)
                return True
        attempts += 1        
        time.sleep(0.5)

    if exflg:
        raise IconNotFoundException("未找到图标,程序退出")
    return False

def predict_icon_status(image, clf, scaler):
    """通过机器学习的模型来验证图标状态"""
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 转换到HSV颜色空间
    hist = cv2.calcHist([image_hsv], [0, 1, 2], None, [16, 16, 16], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist_scaled = scaler.transform([hist.flatten()])
    return clf.predict(hist_scaled)[0] == 1

def preprocess_image(image):
    """预处理图像以提高OCR的准确性。"""
    # 将图像转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 应用CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))
    contrast_enhanced = clahe.apply(gray)

    # 应用锐化
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
    # 二值化
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def find_txt_ocr3(txt, region=None, save_path=None):
    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)

    ocr = CnOcr(rec_model_name='scene-densenet_lite_246-gru_base')
    screen_image = capture_screen_area(region)

    # 对截取的图像进行预处理
    processed_image = preprocess_image(screen_image)
    # 如果提供了保存路径，则保存预处理后的图像
    if save_path:
        cv2.imwrite(save_path, processed_image)  # 保存预处理后的图像为文件

    res = ocr.ocr(processed_image)  # 对预处理后的图像应用OCR
    for line in res:
        if txt in line['text']:
            x = region[0] + line['position'][0][0] + (line['position'][1][0] - line['position'][0][0]) // 2
            y = region[1] + line['position'][0][1] + (line['position'][2][1] - line['position'][0][1]) // 2
            print(f"Interacted with text {txt} at position ({x}, {y}).")
            return line['text']

    print(f"{txt} not found, retrying...")


def main():
    
    time.sleep(3)

    #代理人对话窗口
    """
    agent_panel3 = screen_regions['agent_panel3']
     
    try:
          goods=get_goods(agent_panel3)
          print(f"1.2 获得货物:{goods}")
          pyautogui.hotkey('ctrl', 'w')
    except GoodsNotFoundException as e:
          print(e)

    """

    clf_chakanrenwu1, scaler_chakanrenwu1 = models['chakanrenwu1']
    template_chakanrenwu1, w_chakanrenwu1, h_chakanrenwu1 = templates['chakanrenwu1']
     
    clf_yunshumubiao1, scaler_yunshumubiao1 = models['yunshumubiao1']
    template_yunshumubiao1, w_yunshumubiao1, h_yunshumubiao1 = templates['yunshumubiao1']

    #代理人对话窗口
    agent_panel3 = screen_regions['agent_panel3']

    #"""
    # 当有查看任务时

    if find_icon(template_chakanrenwu1, w_chakanrenwu1, h_chakanrenwu1, clf_chakanrenwu1, scaler_chakanrenwu1,2):
        #print("2.1 查看任务")
        pyautogui.leftClick()
        time.sleep(2)
        try:
          goods=get_goods(agent_panel3)
          print(f"2.2 获得货物:{goods}")
          pyautogui.hotkey('ctrl', 'w')
        except GoodsNotFoundException as e:
          print(e)

    #"""

if __name__ == "__main__":
    main()