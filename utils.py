import cv2
import pyautogui
import numpy as np
from joblib import load
from cnocr import CnOcr
import time
import json
import re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import logging
from say import speak

# 禁用cnocr和cnstd的use model日志
class NoCnocrFilter(logging.Filter):
    def filter(self, record):
        return not ('use model' in record.getMessage().lower())
logging.getLogger('').addFilter(NoCnocrFilter())
logging.getLogger('cnocr').setLevel(logging.ERROR)
logging.getLogger('cnstd').setLevel(logging.ERROR)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('task.log', mode='a'),
        logging.StreamHandler()
    ],
    force=True
)

# 调试截图目录
DEBUG_DIR = "debug_screenshots"
if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

def adjust_region(region, base_resolution=(1920, 1080)):
    """根据当前屏幕分辨率调整区域坐标"""
    current_width, current_height = pyautogui.size()
    base_width, base_height = base_resolution
    x, y, w, h = region
    x = int(x * current_width / base_width)
    y = int(y * current_height / base_height)
    w = int(w * current_width / base_width)
    h = int(h * current_height / base_height)
    return (x, y, w, h)

def log_message(level, message, screenshot=False, region=None):
    """通用日志函数，仅记录关键信息"""
    try:
        if level == "INFO":
            logging.info(message)
        elif level == "ERROR":
            logging.error(message)
        elif level == "WARNING":
            logging.warning(message)
        
        # 临时关闭保存截图功能
        # if screenshot:
        #     screenshot_path = save_screenshot("utils", f"_{level.lower()}_{int(time.time())}")
        
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
    except Exception as e:
        print(f"日志记录错误: {e}")

def save_screenshot(script_name, suffix=""):
    """保存全屏截图"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    screenshot_path = os.path.join(DEBUG_DIR, f"{timestamp}_{script_name}{suffix}.png")
    pyautogui.screenshot().save(screenshot_path)
    return screenshot_path

class IconNotFoundException(Exception):
    pass

class TextNotFoundException(Exception):
    pass

# 屏幕区域配置
screen_regions = {
    'full_right_panel': (1380, 30, 540, 1000),
    'upper_right_panel': (1380, 30, 540, 260),
    'upper_left_panel': (0, 0, 400, 200),
    'mid_left_panel': (50, 150, 500, 600),
    'agent_panel1': (1450, 250, 500, 500),
    'agent_panel2': (1500, 400, 400, 500),
    'agent_panel3': (200, 100, 1400, 900),
    'cangku_panel3': (0, 0, 1700, 850),
    'need_goods_panel': (50, 50, 400, 500)
}

def scollscreen():
    """水平转动屏幕"""
    time.sleep(0.2)
    fx, fy = pyautogui.size()
    pyautogui.moveTo(250, 700)
    pyautogui.dragRel(-30, 0, 0.5, pyautogui.easeOutQuad)

def capture_screen_area(region, save_path=None):
    """捕获屏幕区域截图并转换为OpenCV格式"""
    adjusted_region = adjust_region(region)
    screenshot = pyautogui.screenshot(region=adjusted_region)
    if save_path:
        screenshot.save(save_path)
    screen_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screen_image

def preprocess_image(image):
    """预处理图像以提高OCR准确性"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))
    contrast_enhanced = clahe.apply(gray)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def check_errors():
    """检查错误提示窗口"""
    if find_txt_ocr("错误", max_attempts=1, region=None) or find_txt_ocr("警告", max_attempts=1, region=None):
        log_message("WARNING", "检测到错误或警告提示，尝试关闭窗口", screenshot=True)
        return True
    return False

def run_with_timeout(timeout, action, log_prefix):
    """带超时的循环执行"""
    start_time = time.time()
    while time.time() < start_time + timeout:
        if action():
            return True
        time.sleep(1)
    log_message("ERROR", f"{log_prefix}超时", screenshot=True)
    return False


def find_icon_cnn(icon, region, offset_x, offset_y, threshold, cnn_threshold):
    """
    使用训练好的CNN模型单次查找屏幕上的图标。

    参数:
        icon (str): 图标名称
        region (tuple): 搜索区域 (x, y, width, height)，若为None则全屏搜索
        offset_x (int): X轴偏移量
        offset_y (int): Y轴偏移量
        threshold (float): 模板匹配阈值
        cnn_threshold (float): CNN置信度阈值

    返回:
        bool: 是否找到并移动鼠标到图标位置
    """
    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)
    else:
        region = adjust_region(region)

    # 加载模板图像
    template_path = os.path.join('icon', f"{icon}-1.png")
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        log_message("ERROR", f"模板图像 '{template_path}' 无法加载", screenshot=True)
        raise FileNotFoundError(f"模板图像 '{template_path}' 无法加载")
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    icon_height, icon_width = template.shape[:2]
    w, h = template_gray.shape[::-1]

    # 定义CNN模型结构，与训练时一致
    class IconCNN(nn.Module):
        def __init__(self):
            super(IconCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
            self.fc1 = nn.Linear(128 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, 2)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)

        def forward(self, x):
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = self.pool(self.relu(self.bn2(self.conv2(x))))
            x = self.pool(self.relu(self.bn3(self.conv3(x))))
            x = self.adaptive_pool(x)
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IconCNN().to(device)
    model_path = os.path.join('model_cnn', f"{icon}_classifier.pth")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        log_message("INFO", f"成功加载模型: {model_path}")
    except FileNotFoundError:
        log_message("ERROR", f"模型文件 '{model_path}' 未找到", screenshot=True)
        raise FileNotFoundError(f"模型文件 '{model_path}' 未找到")
    except RuntimeError as e:
        log_message("ERROR", f"加载模型失败: {e}", screenshot=True)
        raise RuntimeError(f"加载模型失败: {e}. 请确保模型结构与训练时一致")
    model.eval()

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 设置分类类别
    opposite_icon = f"{icon.split('-')[0]}-0"
    if os.path.exists(os.path.join('traindata', opposite_icon)):
        class_names = [icon, opposite_icon]
    else:
        class_names = [icon, 'other']

    # 单次模板匹配
    screen_image = capture_screen_area(region)
    screen_gray = cv2.cvtColor(screen_image, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    max_val = np.max(res)
    log_message("INFO", f"模板匹配最大值 = {max_val:.4f}")
    loc = np.where(res >= threshold)
    matches = list(zip(*loc[::-1]))

    if matches:
        for idx, pt in enumerate(matches):
            top_left = pt
            bottom_right = (top_left[0] + w, top_left[1] + h)
            icon_image = screen_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            debug_path = os.path.join(DEBUG_DIR, f"{icon}_match_{idx}.png")
            cv2.imwrite(debug_path, icon_image)
            log_message("INFO", f"保存候选区域图像: {debug_path}")
            icon_rgb = cv2.cvtColor(icon_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(icon_rgb)
            tensor = transform(pil_image).unsqueeze(0).to(device)

            # CNN 分类
            with torch.no_grad():
                output = model(tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_class = class_names[predicted.item()]
                log_message("INFO", f"图标 {icon} 在 {pt} 的置信度: {confidence.item():.4f}, 预测: {predicted_class}")
                if predicted.item() == 1 and confidence.item() > cnn_threshold:
                    x = top_left[0] + region[0] + w // 2 + offset_x
                    y = top_left[1] + region[1] + h // 2 + offset_y
                    pyautogui.moveTo(x, y)
                    log_message("INFO", f"找到图标 {icon}，坐标: ({x}, {y})")
                    return True

    log_message("WARNING", f"图标 {icon} 未找到", screenshot=True)
    return False

def safe_find_icon(icon, region, max_attempts=1, action="click", offset_x=0, offset_y=0, threshold=0.85, cnn_threshold=0.8):
    """封装图标查找和操作，支持多次尝试"""
    attempts = 0
    while attempts < max_attempts:
        if find_icon_cnn(icon, region, offset_x, offset_y, threshold, cnn_threshold):
            time.sleep(0.5)
            if action == "click":
                pyautogui.leftClick()
            log_message("INFO", f"找到[{icon}]并执行{action}", screenshot=True)
            return True
        time.sleep(1)
        scollscreen()
        attempts += 1
    log_message("ERROR", f"未找到[{icon}]，尝试次数: {max_attempts}", screenshot=True)
    return False

def hscollscreen():
    """垂直滑动屏幕"""
    pyautogui.moveTo(1600, 400)
    pyautogui.scroll(-2000)
    time.sleep(0.2)

def rolljump(max_attempts=0):
    """循环跳跃星门"""
    region_full_right = screen_regions['full_right_panel']
    attempts = 0
    while max_attempts == 0 or attempts < max_attempts:
        if safe_find_icon("jump3", region_full_right, max_attempts=1):
            # time.sleep(0.2)
            # pyautogui.click()
            # time.sleep(10)
            # pyautogui.click()
            log_message("INFO", "找到jump3，退出rolljump")
            # speak("找到jump3，退出rolljump")
            return True
        else:
            safe_find_icon("jump1", region_full_right, max_attempts=1)
            safe_find_icon("jump2", region_full_right, max_attempts=1)          
            log_message("INFO", "rolljump,循环跳跃星门:{attempts}")
            # pyautogui.click()
        time.sleep(2)
        attempts += 1
    log_message("ERROR", f"rolljump达到最大尝试次数: {max_attempts}", screenshot=True)
    return False


def find_txt_ocr(txt, max_attempts=5, region=None):
    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)
    else:
        region = adjust_region(region)

    attempts = 0
    while attempts < max_attempts:
        ocr = CnOcr(rec_model_name='scene-densenet_lite_246-gru_base')
        screen_image = pyautogui.screenshot(region=region)
        res = ocr.ocr(screen_image)

        for line in res:
            if txt in line['text']:
                x = region[0] + line['position'][0][0] + (line['position'][1][0] - line['position'][0][0]) // 2
                y = region[1] + line['position'][0][1] + (line['position'][2][1] - line['position'][0][1]) // 2
                pyautogui.moveTo(x, y)
                log_message("INFO", f"找到文本: {txt}，坐标: ({x}, {y})")
                return True

        time.sleep(0.5)
        attempts += 1
        scollscreen()

    log_message("ERROR", f"文本: {txt} 查找失败，尝试次数: {max_attempts}", screenshot=True)
    return False

def find_txt_ocr2(txt, max_attempts=5, region=None):
    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)
    else:
        region = adjust_region(region)

    attempts = 0
    while attempts < max_attempts:
        ocr = CnOcr(rec_model_name='scene-densenet_lite_246-gru_base')
        screen_image = capture_screen_area(region, save_path=f"debug_image_before.png")
        save_path = "debug_image_after.png"
        if save_path:
            cv2.imwrite(save_path, screen_image)

        res = ocr.ocr(screen_image)
        extracted_goods_name = None
        pattern = re.compile(f'{txt}([^()]+)')
        
        for item in res:
            match = pattern.search(item['text'])
            if match:
                extracted_goods_name = re.sub(r'[^\u4e00-\u9fff]', '', match.group(1))
                log_message("INFO", f"OCR识别到货物名称: {extracted_goods_name}")
                return extracted_goods_name
        
        attempts += 1
        time.sleep(0.5)
        scollscreen()

    log_message("ERROR", f"文本: {txt} 查找失败，尝试次数: {max_attempts}", screenshot=True)
    return None

def load_location_name(tag):
    try:
        with open('addr.txt', 'r', encoding='utf-8-sig') as file:
            content = file.read()
            data = json.loads(content)
            result = data.get(tag)
            log_message("INFO", f"加载位置名称: {tag} -> {result}")
            return result
    except FileNotFoundError:
        log_message("ERROR", "addr.txt 文件未找到", screenshot=True)
        print("文件未找到。")
    except json.JSONDecodeError:
        log_message("ERROR", "解析 addr.txt JSON 时出错", screenshot=True)
        print("解析 JSON 时出错。")
    except UnicodeDecodeError:
        log_message("ERROR", "addr.txt 文件编码问题，无法读取", screenshot=True)
        print("文件编码问题，无法读取。")
    return None

def correct_string(input_str):
    rules = [
        ('天', '大'),
        ('性', '牲'),
        ('拉', '垃'),
        ('级', '圾'),
        ('杀', 'OP杀'),
        ('者门', '看门'),
    ]
    for old, new in rules:
        input_str = re.sub(old, new, input_str)
    log_message("INFO", f"字符串校正: {input_str}")
    return input_str

def load_config():
    config = {}
    with open('cfg.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            key, value = line.strip().split('=')
            config[key.strip()] = value.strip()
    log_message("INFO", f"加载配置: {config}")
    return config

def find_and_close_icons(icon, region):
    if safe_find_icon(icon, region, max_attempts=1, threshold=0.86):
        pyautogui.click()
        time.sleep(1)
        log_message("INFO", f"关闭图标: {icon}")
        return True
    log_message("INFO", f"未找到关闭图标: {icon}")
    return False

def close_icons_main():
    find_and_close_icons("close1", region=None)