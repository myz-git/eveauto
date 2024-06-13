import cv2
import numpy as np
import pyautogui
import pytesseract
import pyperclip  # 导入 pyperclip
import time
from joblib import load
# 在 talk.py 中添加对 close.py 的调用
from close import close_icons_main
from outsite import outsite_icons_main

import os
import json
from cnocr import CnOcr

import re

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


def extract_goods_name(text):
    """从文本中提取指定格式的商品名称"""
    pattern = re.compile(r'货物\d*×(.+?)\*')
    match = pattern.search(text)
    if match:
        goods_name = match.group(1).strip()  # 提取并去除可能的前后空白
        return goods_name
    return None  # 如果没有匹配到，返回 None

def find_icon(template, width, height, clf, scaler, max_attempts=10,offset_x=0,offset_y=0,region=None,):
    """查找图标,找不到则终止程序"""   
    
    # 如果没有提供 region，使用全屏作为默认区域
    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)

    attempts = 0
    while attempts < max_attempts:        
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
                #当使用capture_screen_area后需要使用全局坐标,即region(x,y) ,即region[0] ,region[1]
                x=max_loc[0] + region[0] + width // 2 + offset_x
                y=max_loc[1] + region[1] + height // 2 + offset_y
                pyautogui.moveTo(x, y)
                print(f"Icon detected!")
                return True
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts}: not found, Retrying...")
        scollscreen()  
        time.sleep(1)
    
    print(f"Icon Not found after maximum attempts. 程序终止...")
    # 程序退出
    exit(1)

def find_icon_noexit(template, width, height, clf, scaler, max_attempts=10,offset_x=0,offset_y=0,region=None,):
    """查找图标,即使找不到也不退出程序"""      
    # 如果没有提供 region，使用全屏作为默认区域
    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)
    attempts = 0
    while attempts < max_attempts:        
        screen = capture_screen_area(region)
        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_screen, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.8:
            icon_image = screen[max_loc[1]:max_loc[1]+height, max_loc[0]:max_loc[0]+width]
            if predict_icon_status(icon_image, clf, scaler):
                #当使用capture_screen_area后需要使用全局坐标,即region(x,y) ,即region[0] ,region[1]
                x=max_loc[0] + region[0] + width // 2 + offset_x
                y=max_loc[1] + region[1] + height // 2 + offset_y
                pyautogui.moveTo(x, y)
                return True
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts}: not found, Retrying...")
        scollscreen()  
        time.sleep(1)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 应用自适应阈值
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return processed


def find_txt_ocr(txt,  max_attempts=10, region=None):
    """使用OCR在屏幕特定区域查找"""
    attempts = 0
    while attempts < max_attempts:        
        
       
        # 初始化OCR工具
        ocr = CnOcr()
        screen_image=pyautogui.screenshot(region=region)
        res = ocr.ocr(screen_image)  # 使用 ocr 方法处理整个图像
        # 执行OCR


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
                
                print(f"Interacted with agent {txt} at position ({x}, {y}).")
                return True
        #print(data)  # 打印所有识别到的文本，看是否包括目标文本
        time.sleep(1)
        attempts += 1
        scollscreen()
        print(f"Attempt {attempts}/{max_attempts}: {txt} not found, retrying...")
        

    print(f"{txt} not found after maximum attempts. Exiting...")
    exit(1)

def find_txt_ocr2(txt,  max_attempts=10, region=None):
    """使用CNOCR高识别率模型,在屏幕特定区域查找"""
    attempts = 0
    while attempts < max_attempts:                
        # 初始化OCR工具
        ocr = CnOcr()
        #ocr = CnOcr(rec_model_name='densenet_lite_246-gru_base',context='GPU')
        # 执行OCR
        screen_image=pyautogui.screenshot(region=region)
        res = ocr.ocr(screen_image)  # 使用 ocr 方法处理整个图像

        # 打印OCR结果
        print("OCR results:", res)

        # 遍历每一行的识别结果
        extracted_goods_name=None

        pattern = re.compile(f'{txt}([^()]+)')
        # 遍历所有字典
        for item in res:
            match = pattern.search(item['text'])
            if match:
                # 找到固定部分的起始位置
                extracted_goods_name = re.sub(r'[^\u4e00-\u9fff]', '', match.group(1))
                print(f"货物名称: {extracted_goods_name}")
                return extracted_goods_name
    

        time.sleep(1)
        attempts += 1
        scollscreen()
        print(f"Attempt {attempts}/{max_attempts}: {txt} not found, retrying...")
        

    print(f"{txt} not found after maximum attempts. Exiting...")
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

    #运输目标
    clf_yunshumubiao1 = load('model/trained_model_yunshumubiao1.joblib')
    scaler_yunshumubiao1 = load('model/scaler_yunshumubiao1.joblib')

    #仓库搜索
    clf_search2 = load('model/trained_model_search2.joblib')
    scaler_search2 = load('model/scaler_search2.joblib')

    #机库
    clf_jiku1 = load('model/trained_model_jiku1.joblib')
    scaler_jiku1 = load('model/scaler_jiku1.joblib')

    #仓库
    clf_jiku2 = load('model/trained_model_jiku2.joblib')
    scaler_jiku2 = load('model/scaler_jiku2.joblib')
    
    #查看任务
    clf_chakanrenwu1 = load('model/trained_model_chakanrenwu1.joblib')
    scaler_chakanrenwu1 = load('model/scaler_chakanrenwu1.joblib')


    # 构建图标路径
    icon_path_agent1 = os.path.join('icon', 'agent1-1.png')
    icon_path_agent2 = os.path.join('icon', 'agent2-1.png')
    icon_path_agent3 = os.path.join('icon', 'agent3-1.png')
    icon_path_yunshumubiao1 = os.path.join('icon', 'yunshumubiao1-1.png')
    icon_path_search2 = os.path.join('icon', 'search2-1.png')
    icon_path_jiku1 = os.path.join('icon', 'jiku1-1.png')
    icon_path_jiku2 = os.path.join('icon', 'jiku2-1.png')
    icon_path_chakanrenwu1 = os.path.join('icon', 'chakanrenwu1-1.png')
    
    #
    template_agent1 = cv2.imread(icon_path_agent1, cv2.IMREAD_COLOR)
    template_agent2 = cv2.imread(icon_path_agent2, cv2.IMREAD_COLOR)
    template_agent3 = cv2.imread(icon_path_agent3, cv2.IMREAD_COLOR)
    template_yunshumubiao1 = cv2.imread(icon_path_yunshumubiao1, cv2.IMREAD_COLOR)
    template_search2 = cv2.imread(icon_path_search2, cv2.IMREAD_COLOR)
    template_jiku1 = cv2.imread(icon_path_jiku1, cv2.IMREAD_COLOR)
    template_jiku2 = cv2.imread(icon_path_jiku2, cv2.IMREAD_COLOR)
    template_chakanrenwu1 = cv2.imread(icon_path_chakanrenwu1, cv2.IMREAD_COLOR)

    #
    template_gray_agent1 = cv2.cvtColor(template_agent1, cv2.COLOR_BGR2GRAY)
    template_gray_agent2 = cv2.cvtColor(template_agent2, cv2.COLOR_BGR2GRAY)
    template_gray_agent3 = cv2.cvtColor(template_agent3, cv2.COLOR_BGR2GRAY)
    template_gray_yunshumubiao1 = cv2.cvtColor(template_yunshumubiao1, cv2.COLOR_BGR2GRAY)
    template_gray_search2 = cv2.cvtColor(template_search2, cv2.COLOR_BGR2GRAY)
    template_gray_jiku1 = cv2.cvtColor(template_jiku1, cv2.COLOR_BGR2GRAY)
    template_gray_jiku2 = cv2.cvtColor(template_jiku2, cv2.COLOR_BGR2GRAY)
    template_gray_chakanrenwu1 = cv2.cvtColor(template_chakanrenwu1, cv2.COLOR_BGR2GRAY)

    #
    w_agent1, h_agent1 = template_gray_agent1.shape[::-1]
    w_agent2, h_agent2 = template_gray_agent2.shape[::-1]
    w_agent3, h_agent3 = template_gray_agent3.shape[::-1]
    w_yunshumubiao1, h_yunshumubiao1 = template_gray_yunshumubiao1.shape[::-1]
    w_search2, h_search2 = template_gray_search2.shape[::-1]
    w_jiku1, h_jiku1 = template_gray_jiku1.shape[::-1]
    w_jiku2, h_jiku2 = template_gray_jiku2.shape[::-1]
    w_chakanrenwu1, h_chakanrenwu1 = template_gray_chakanrenwu1.shape[::-1]

    #设置需要捕获的屏幕区域
    x0, y0, width0, height0 = 1450, 250, 500, 500
    region0=(x0,y0,width0,height0)

    #代理人列表窗口
    x1, y1, width1, height1 = 1500, 400, 400, 500
    region1=(x1,y1,width1,height1)

    #代理人对话窗口
    x2, y2, width2, height2 = 350, 100, 850, 850
    region2=(x2,y2,width2,height2)

    #仓库
    x3, y3, width3, height3 = 0, 0, 1500, 850
    region3=(x3,y3,width3,height3)

    # 1. 准备开始
    time.sleep(1)

    # 2. 查找"代理人"图标    
    if find_icon(template_gray_agent1, w_agent1, h_agent1, clf_agent1, scaler_agent1,10,0,0,region0):
        pyautogui.leftClick()
    
    
    # 3. 查找代理人
    # 3.1 获得代理人名字
    agent_name = load_location_name('agent')
    print(f"agent={agent_name}")
    # 3.2 通过OCR文字识别查找代理人
    if find_txt_ocr(agent_name,5,region1):
        pyautogui.hotkey('ctrl', 'w')
        pyautogui.doubleClick()  # 双击打开代理人对话窗口
        time.sleep(1)

    # 4.1 和代理人对话[查看任务](之前点过接受任务)
    if find_icon_noexit(template_gray_chakanrenwu1, w_chakanrenwu1, h_chakanrenwu1, clf_chakanrenwu1, scaler_chakanrenwu1,3):
        print("查看新任务")
        pyautogui.leftClick()
        time.sleep(1)
    else:
        # 4.2 和代理人对话接任务-[我要执行新任务]
        if find_icon_noexit(template_gray_agent2, w_agent2, h_agent2, clf_agent2, scaler_agent2,3): 
            print("我要执行新任务")
            pyautogui.leftClick()
            time.sleep(1)

        # 5. 和代理人对话接任务-"接受任务"
        if find_icon_noexit(template_gray_agent3, w_agent3, h_agent3, clf_agent3, scaler_agent3,3):
            pyautogui.leftClick()
            time.sleep(1)

    # 6. 代理人对话中,定位运输目标
    goods=None
    if find_icon(template_gray_yunshumubiao1, w_yunshumubiao1, h_yunshumubiao1, clf_yunshumubiao1, scaler_yunshumubiao1,5,0,0,region2):
        print("查找运输目标")        
        time.sleep(1)
        # 获取货物内容, 根据运输目标坐标直接定位货物右侧的OCR扫描范围
        x, y = pyautogui.position()         
        region_hw=(x+80,y+94,300,150)
        
        #只匹配汉字, 如果折行只提取第一行
        goods=find_txt_ocr2('',3,region_hw)
        if goods is not None:
            pyautogui.hotkey('ctrl', 'w')
        else:
            print(f"未取得运输目标,程序终止!")
            exit(1)

    
    # 7. 从仓库搬运货物到舰船机库
    # 打开仓库
    pyautogui.hotkey('alt', 'c')
    
    # 获取机库坐标   
    find_icon(template_gray_jiku1, w_jiku1, h_jiku1, clf_jiku1, scaler_jiku1,5,15,15,region3)
    jiku_x,jiku_y = pyautogui.position() 

    # 激活仓库   
    if find_icon(template_gray_jiku2, w_jiku2, h_jiku2, clf_jiku2, scaler_jiku2,5,0,0,region3):
        pyautogui.leftClick()
        time.sleep(0.2)

    # 搜索仓库
    if find_icon(template_gray_search2, w_search2, h_search2, clf_search2, scaler_search2,5,0,0,region3):
        print("搜索仓库")
        pyautogui.leftClick()
        pyperclip.copy(goods)  # 复制名称到剪贴板
        pyautogui.hotkey('ctrl', 'v')  # 粘贴名称
        pyautogui.press('enter')
        time.sleep(1)
                
        # 移动仓库货物到机库
        pyautogui.moveRel(0,65)
        time.sleep(1)
        pyautogui.dragTo(jiku_x,jiku_y,1,pyautogui.easeOutQuad)
        print(f"[{goods}]已放入机库中,准备送货...")
    
    
    close_icons_main()
    pyautogui.hotkey('ctrl', 'w')
    #出站
    outsite_icons_main()
    print("出站中,请等待...")
    time.sleep(10)

    print("Program ended.")
    

if __name__ == "__main__":
    main()
