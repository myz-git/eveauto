import cv2
import numpy as np
import pyautogui
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import sys
import os

# 定义 CNN 模型（与训练时一致）
class IconCNN(torch.nn.Module):
    def __init__(self):
        super(IconCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(128 * 4 * 3, 256)
        self.fc2 = torch.nn.Linear(256, 2)  # 2 类：jump1-1、jump1-0
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 3)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 加载训练好的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IconCNN().to(device)
try:
    model.load_state_dict(torch.load('icon_classifier.pth'))
except FileNotFoundError:
    print("Error: icon_classifier.pth not found. Please ensure the model file exists.")
    sys.exit(1)
except RuntimeError as e:
    print(f"Error loading model: {e}")
    print("The model in 'icon_classifier.pth' may have a different number of classes. Please retrain using 'train.py'.")
    sys.exit(1)
model.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 24)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载模板图标（jump1-1 的模板）
template_path = os.path.join('icon', 'jump1-1.png')
template = cv2.imread(template_path, cv2.IMREAD_COLOR)
if template is None:
    raise FileNotFoundError(f"模板图像 '{template_path}' 无法加载")
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

def capture_screen():
    """捕获整个屏幕截图并返回，转换为 BGR 格式。"""
    screenshot = pyautogui.screenshot()
    screen_image = np.array(screenshot)
    screen_image = cv2.cvtColor(screen_image, cv2.COLOR_RGB2BGR)
    return screen_image

def sliding_window(image, step_size, window_size):
    """滑动窗口遍历图像，返回窗口区域和坐标。"""
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def color_filter(window, x, y):
    """改进颜色过滤，检查窗口区域是否接近白色（jump1-1 的特征）。"""
    hsv = cv2.cvtColor(window, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    mean_brightness = np.mean(v_channel)
    s_channel = hsv[:, :, 1]
    mean_saturation = np.mean(s_channel)
    
    # 放宽条件：亮度 > 180，饱和度 < 50
    if mean_brightness < 180 or mean_saturation > 50:
        print(f"坐标 ({x}, {y}) 被颜色过滤: 亮度={mean_brightness:.2f}, 饱和度={mean_saturation:.2f}")
        return False
    return True

def template_match_filter(window, x, y):
    """使用模板匹配筛选外形，检查窗口区域是否与 jump1-1 模板相似。"""
    window_gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    
    # 模板匹配
    res = cv2.matchTemplate(window_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    
    # 降低阈值：匹配度 > 0.4
    if max_val > 0.4:
        print(f"坐标 ({x}, {y}) 模板匹配通过: 匹配度={max_val:.4f}")
        return True
    else:
        print(f"坐标 ({x}, {y}) 被模板匹配过滤: 匹配度={max_val:.4f}")
        return False

def detect_icon(search_interval=1, num_searches=100):
    """使用模板匹配和 CNN 模型在屏幕上搜索 jump1-1，返回坐标 (x, y)。"""
    window_size = (32, 24)  # 图标大小
    step_size = 2  # 减小步长到 2
    class_names = ['jump1-0', 'jump1-1']  # 类别名称

    for _ in range(num_searches):
        start_time = time.time()  # 记录搜索开始时间
        
        screen_image = capture_screen()
        
        # 使用滑动窗口遍历屏幕
        for x, y, window in sliding_window(screen_image, step_size, window_size):
            # 颜色过滤
            if not color_filter(window, x, y):
                continue

            # 模板匹配筛选外形
            if not template_match_filter(window, x, y):
                continue

            # 转换为 PIL 图像并预处理
            window_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(window_rgb)
            tensor = transform(pil_image).unsqueeze(0).to(device)

            # 使用 CNN 模型推理
            with torch.no_grad():
                output = model(tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                # 调试：打印置信度和预测类别
                predicted_class = class_names[predicted.item()]
                print(f"坐标 ({x}, {y}) 预测类别: {predicted_class}, 置信度: {confidence.item():.4f}")

                # 放宽条件：置信度 > 0.9
                if predicted.item() == 1 and confidence.item() > 0.9:  # jump1-1 类别为 1
                    print(f"找到 jump1-1 坐标: ({x}, {y})")
                    cv2.imshow("Detected jump1-1", window)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    return (x, y)

        end_time = time.time()  # 记录搜索结束时间
        search_time = end_time - start_time
        print(f"未找到 jump1-1，单次搜索耗时: {search_time:.3f} 秒")
        time.sleep(search_interval)

    return None

if __name__ == "__main__":
    result = detect_icon()
    if result:
        x, y = result
        print(f"最终找到 jump1-1 坐标: ({x}, {y})")
    else:
        print("搜索完成，未找到 jump1-1")