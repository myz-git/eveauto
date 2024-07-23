# utils.py
import cv2
import pyautogui
import numpy as np
from joblib import load
from cnocr import CnOcr
import time
import json
import re

class IconNotFoundException(Exception):
    """Exception raised when an icon is not found."""
    pass

class TextNotFoundException(Exception):
    """Exception raised when the specified text is not found."""
    pass

def scollscreen():
    """水平转动屏幕"""
    time.sleep(0.2)
    fx, fy = pyautogui.size()
    pyautogui.moveTo(100, 600)
    pyautogui.dragRel(-30, 0, 0.5, pyautogui.easeOutQuad)


def capture_screen_area(region, save_path=None):
    """捕获屏幕上指定区域的截图并转换为OpenCV格式，可选地保存到文件"""
    screenshot = pyautogui.screenshot(region=region)
    if save_path:
        screenshot.save(save_path)  # 保存截图到指定路径
    screen_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screen_image

def preprocess_image(image):
    """预处理图像以提高OCR的准确性。
    1. 调整CLAHE参数
    CLAHE（对比度限制的自适应直方图均衡）有两个主要的参数可以调整：

    clipLimit：这是对比度限制的阈值，较高的值会更显著地增强对比度。如果图像的对比度已经很高，较低的值可能更有效。
    tileGridSize：这是定义了将图像分割为多少块来分别应用直方图均衡。更小的网格尺寸会对局部细节有更好的增强效果，但可能会导致图像过度锐化或噪声增加。
    2. 锐化参数
    锐化过程中使用的核（kernel）影响图像的锐化程度。可以调整核矩阵来改变锐化强度：

    锐化核的强度可以通过调整中心值和周围负值的大小来改变。例如，增加中心值会增加锐化效果。
    3. 二值化阈值
    二值化过程通过阈值将图像转换为黑白两色，对于OCR识别来说非常关键：

    阈值选择：如果使用的是OTSU方法，它会自动选择一个最佳阈值。如果结果不理想，可以尝试手动设置固定的阈值，或者使用自适应阈值方法（如cv2.adaptiveThreshold），它可以根据图像各区域的亮度变化自动调整阈值。
    4. 高斯模糊或中值模糊
    在二值化之前应用模糊可以帮助去除噪声，但如果过度模糊可能会使得文字边界模糊不清。可以尝试调整：

    高斯模糊的核大小和标准差。
    中值模糊的核大小，通常用于去除椒盐噪声。
    """

    # 将图像转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 应用CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))  # 16,32比较清晰
    contrast_enhanced = clahe.apply(gray)

    #注: 这里不能使用高斯模糊,不能正常显示文字

    # 应用锐化
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
    # 二值化
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def predict_icon_status(image, clf, scaler):
    """通过机器学习的模型来验证图标状态"""
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 转换到HSV颜色空间
    hist = cv2.calcHist([image_hsv], [0, 1, 2], None, [16, 16, 16], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist_scaled = scaler.transform([hist.flatten()])
    return clf.predict(hist_scaled)[0] == 1

def load_model_and_scaler(model_path):
    """加载模型和标准器"""
    clf = load(f'{model_path}.joblib')
    scaler = load(f'{model_path}_scaler.joblib')
    return clf, scaler

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
        scollscreen()

    if exflg:
        raise IconNotFoundException("未找到图标,程序退出")
    return False

def find_txt_ocr(txt,  max_attempts=5, region=None):
    """使用OCR在屏幕特定区域查找"""

    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)

    attempts = 0
    while attempts < max_attempts:        
        # 初始化OCR工具
        #ocr = CnOcr()
        ocr = CnOcr(rec_model_name='scene-densenet_lite_246-gru_base')
        screen_image=pyautogui.screenshot(region=region)
        res = ocr.ocr(screen_image)  # 使用 ocr 方法处理整个图像
        # 打印OCR结果
        #print("OCR results:", res)

        # 遍历每一行的识别结果
        for line in res:
            if txt in line['text']:
                # 假设我们可以获取到文字的位置
                x = region[0] + line['position'][0][0] + (line['position'][1][0] - line['position'][0][0]) // 2
                y = region[1] + line['position'][0][1] + (line['position'][2][1] - line['position'][0][1]) // 2
            
                # 移动鼠标并点击文字
                pyautogui.moveTo(x, y)
                
                print(f"Interacted with agent {txt} at position ({x}, {y}).")
                return True
        #print(data)  # 打印所有识别到的文字，看是否包括目标文字
        time.sleep(0.5)
        attempts += 1
        scollscreen()
        print(f"Attempt {attempts}/{max_attempts}: {txt} not found, retrying...")
        
    return False
    

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
        #保存识别前截取的图片范围   
        screen_image = capture_screen_area(region, save_path=f"debug_image_before.png")  # Save each attempt

        # 对截取的图像进行预处理(二值化,增加对比度,锐化)
        #screen_image = preprocess_image(screen_image)

        # 如果提供了保存路径，则保存预处理后的图像
        save_path="debug_image_after.png"
        if save_path:
            cv2.imwrite(save_path, screen_image)  # 保存预处理后的图像为文件

        res = ocr.ocr(screen_image)  # 对预处理后的图像应用OCR
        # 遍历每一行的识别结果
        extracted_goods_name=None
        pattern = re.compile(f'{txt}([^()]+)')
        
        # 遍历所有字典,找出匹配{txt}的文字
        for item in res:
            match = pattern.search(item['text'])
            if match:
                # 找到固定部分的起始位置,只匹配汉字!!
                extracted_goods_name = re.sub(r'[^\u4e00-\u9fff]', '', match.group(1))
                print(f"识别文字: {extracted_goods_name}")
                return extracted_goods_name        
        attempts += 1
        time.sleep(0.5)
        scollscreen()
    print(f"Attempt {attempts}/{max_attempts}: {txt} not found, retrying...")
        


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

def correct_string(input_str):
    # 对OCR货物识别易错的结果进行二次修证
    rules = [
        ('天', '大'),    # 将 '天' 替换为 '大'
        ('性', '牲'),    # 将 '性' 替换为 '牲'
        ('拉', '垃'),
        ('级', '圾'),
        # 可以根据需要添加更多规则
    ]
    
    # 应用每个规则进行替换
    for old, new in rules:
        input_str = re.sub(old, new, input_str)
    
    return input_str