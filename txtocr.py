import cv2
import numpy as np
import pyautogui
import time
import os
from joblib import load

def capture_screen_area(x, y, width, height):
    """捕获屏幕上指定区域的截图并转换为OpenCV格式"""
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    screen_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screen_image

def calculate_histogram(image):
    """计算图像的归一化彩色直方图作为特征向量"""
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def predict_icon_status(image, clf, scaler):
    """使用机器学习模型判断图标状态"""
    hist = calculate_histogram(image)
    hist_scaled = scaler.transform([hist])
    return clf.predict(hist_scaled)[0]

# 加载模型和标准化器
clf = load('trained_model.joblib')
scaler = load('scaler.joblib')

icon_paths = ['jump1-1.png']  # 示例只使用一个图标路径
icons = [cv2.imread(path, cv2.IMREAD_COLOR) for path in icon_paths if cv2.imread(path) is not None]

x1, y1, width1, height1 = 1300, 200, 300, 200

while True:
    panel_image = capture_screen_area(x1, y1, width1, height1)
    for icon in icons:
        if predict_icon_status(panel_image, clf, scaler) == 1:
            print("图标可用")
        else:
            print("图标不可用")

    time.sleep(3)  # 每三秒检查一次
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
