import cv2
import numpy as np
import pyautogui
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
import sys
import os
import logging

# 日志配置，与train.py一致
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('task.log', mode='a'),
        logging.StreamHandler()
    ],
    force=True
)

# 从命令行获取目标图标名称
if len(sys.argv) < 2:
    print("Usage: python verify.py <icon_name> (e.g., python verify.py jump0)")
    sys.exit(1)
icon_name = sys.argv[1]  # 例如 "jump0"

# 加载模板图标以获取大小（用于模板匹配）
template_path = os.path.join('icon', f"{icon_name}-1.png")
template = cv2.imread(template_path, cv2.IMREAD_COLOR)
if template is None:
    logging.error(f"Template image {template_path} not found")
    raise FileNotFoundError(f"Template image {template_path} not found")
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template_height, template_width = template.shape[:2]  # 用于模板匹配
w, h = template_gray.shape[::-1]

# 固定模型输入尺寸，与train.py一致
icon_height, icon_width = 64, 64
if template_height != 64 or template_width != 64:
    logging.warning(f"Template {template_path} size ({template_height}x{template_width}) differs from model input (64x64)")

# 定义 CNN 模型
class IconCNN(nn.Module):
    def __init__(self, icon_height=64, icon_width=64):
        super(IconCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        pooled_height = icon_height // 8
        pooled_width = icon_width // 8
        self.fc1 = nn.Linear(128 * pooled_height * pooled_width, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 加载训练好的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IconCNN(icon_height=64, icon_width=64).to(device)
model_path = os.path.join('model_cnn', f"{icon_name}_classifier.pth")
try:
    model.load_state_dict(torch.load(model_path))
except FileNotFoundError:
    logging.error(f"Model file {model_path} not found. Please ensure the model file exists.")
    sys.exit(1)
except RuntimeError as e:
    logging.error(f"Error loading model: {e}")
    print(f"Error loading model: {e}")
    print("The model may have a different structure. Please retrain using 'train.py'.")
    sys.exit(1)
model.eval()

# 数据预处理（CNN 输入）
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def capture_screen():
    """捕获整个屏幕的截图并返回，转换屏幕截图为OpenCV可处理的BGR格式。"""
    screenshot = pyautogui.screenshot()
    screen_image = np.array(screenshot)
    screen_image = cv2.cvtColor(screen_image, cv2.COLOR_RGB2BGR)
    return screen_image

def detect_icon(template_filename, capture_interval=1, num_captures=100):
    """使用模板匹配找到所有匹配区域，并通过 CNN 区分目标图标和负样本，返回坐标 (x, y)。"""
    # 动态确定类别：优先使用 icon_name 和对应的 icon_name-0（如果存在），否则使用 other
    opposite_icon = f"{icon_name.split('-')[0]}-0"  # 例如 jump0-1 -> jump0-0
    if os.path.exists(os.path.join('traindata', opposite_icon)):
        class_names = [icon_name, opposite_icon]  # 例如 ['jump0-1', 'jump0-0']
    else:
        class_names = [icon_name, 'other']  # 例如 ['jump0-1', 'other']

    for _ in range(num_captures):
        start_time = time.time()  # 记录搜索开始时间
        
        screen_image = capture_screen()
        screen_gray = cv2.cvtColor(screen_image, cv2.COLOR_BGR2GRAY)
        
        # 外形模板匹配
        res = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        threshold = 0.86  # 模板匹配阈值
        loc = np.where(res >= threshold)  # 找到所有匹配度高于阈值的区域
        matches = list(zip(*loc[::-1]))  # 获取所有匹配位置

        if matches:  # 如果有匹配区域
            # 对每个匹配区域进行 CNN 预测
            for pt in matches:
                top_left = pt
                bottom_right = (top_left[0] + w, top_left[1] + h)
                
                # 截取图标区域
                icon_image = screen_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                
                # 转换为 PIL 图像并预处理
                icon_rgb = cv2.cvtColor(icon_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(icon_rgb)
                tensor = transform(pil_image).unsqueeze(0).to(device)

                # 使用 CNN 模型推理
                with torch.no_grad():
                    output = model(tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)

                    predicted_class = class_names[predicted.item()]
                    print(f"坐标 ({top_left[0]}, {top_left[1]}) 模板匹配度: {res[top_left[1], top_left[0]]:.4f}, 预测类别: {predicted_class}, 置信度: {confidence.item():.4f}")

                    # 如果是目标图标且置信度 > 0.85
                    if predicted.item() == 0 and confidence.item() > 0.85:  # 目标图标类别为 0
                        print(f"找到 {icon_name} 坐标: ({top_left[0]}, {top_left[1]})")
                        cv2.imshow(f"Detected {icon_name}", icon_image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        return (top_left[0], top_left[1])
            
            print(f"找到 {len(matches)} 个匹配区域，但都不是 {icon_name}（可能是 {class_names[1]}）")
        else:
            print(f"未找到图标，最高匹配度: {np.max(res):.4f}")

        end_time = time.time()  # 记录搜索结束时间
        search_time = end_time - start_time
        print(f"单次搜索耗时: {search_time:.3f} 秒")
        time.sleep(capture_interval)

    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify.py <icon_name> (e.g., python verify.py jump0)")
        sys.exit(1)

    template_filename = f"{sys.argv[1]}-1.png"  # 从命令行参数获取图标文件名
    result = detect_icon(template_filename)
    if result:
        x, y = result
        print(f"最终找到 {icon_name} 坐标: ({x}, {y})")
    else:
        print(f"搜索完成，未找到 {icon_name}")