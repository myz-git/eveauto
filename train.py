import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import sys
import cv2

# 从命令行获取目标图标名称
if len(sys.argv) < 2:
    print("Usage: python train.py <icon_name> (e.g., python train.py jump2-1)")
    sys.exit(1)
icon_name = sys.argv[1]  # 例如 "jump2-1"

# 加载模板文件以获取图标大小
template_path = os.path.join('icon', f"{icon_name}.png")
template = cv2.imread(template_path, cv2.IMREAD_COLOR)
if template is None:
    raise FileNotFoundError(f"模板图像 '{template_path}' 无法加载")
icon_height, icon_width = template.shape[:2]  # 动态获取图标大小

# 定义 CNN 模型
class IconCNN(nn.Module):
    def __init__(self):
        super(IconCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 根据图标大小动态计算全连接层输入维度
        pooled_height = icon_height // 8  # 3 次池化，每次除以 2
        pooled_width = icon_width // 8
        self.fc1 = nn.Linear(128 * pooled_height * pooled_width, 256)
        self.fc2 = nn.Linear(256, 2)  # 2 类：目标图标 vs 负样本
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # 动态展平
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 数据增强和预处理（增强背景变化）
transform = transforms.Compose([
    transforms.Resize((icon_height, icon_width)),  # 动态调整大小
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # 增强亮度、对比度抖动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载数据集
dataset = datasets.ImageFolder('traindata', transform=transform)  # 修复：添加右括号

# 打印所有类别
print("所有类别:", dataset.classes)

# 动态确定类别：优先使用 icon_name 和对应的 icon_name-0（如果存在），否则使用 other
opposite_icon = f"{icon_name.split('-')[0]}-0"  # 例如 jump1-1 -> jump1-0
if opposite_icon in dataset.classes:
    desired_classes = [icon_name, opposite_icon]  # 例如 ['jump1-1', 'jump1-0']
else:
    desired_classes = [icon_name, 'other']  # 例如 ['jump2-1', 'other']
class_to_idx = {cls: idx for idx, cls in enumerate(desired_classes)}
indices = [i for i, (_, label) in enumerate(dataset.samples) if dataset.classes[label] in desired_classes]

# 创建子数据集
subset_dataset = Subset(dataset, indices)

# 重新映射标签
subset_dataset.dataset.class_to_idx = class_to_idx
subset_dataset.dataset.classes = desired_classes
for i in range(len(subset_dataset)):
    sample_path, _ = subset_dataset.dataset.samples[subset_dataset.indices[i]]
    class_name = os.path.basename(os.path.dirname(sample_path))
    subset_dataset.dataset.samples[subset_dataset.indices[i]] = (sample_path, class_to_idx[class_name])

# 划分训练集和验证集
train_size = int(0.8 * len(subset_dataset))
val_size = len(subset_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(subset_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IconCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 降低学习率

# 训练循环（添加早停逻辑）
num_epochs = 20  # 保持 20 次
best_acc = 0.0
patience = 5  # 早停耐心值：连续 5 次准确率不提升则停止
early_stop_counter = 0
best_loss = float('inf')
loss_patience = 5  # 损失早停耐心值
loss_early_stop_counter = 0
loss_threshold = 1e-6  # 损失变化阈值

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    print(f"Validation Accuracy: {val_acc}%")

    # 早停逻辑（基于准确率）
    if val_acc > best_acc:
        best_acc = val_acc
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}: Validation Accuracy has not improved for {patience} epochs.")
            break

    # 早停逻辑（基于损失）
    if best_loss - avg_loss > loss_threshold:
        best_loss = avg_loss
        loss_early_stop_counter = 0
    else:
        loss_early_stop_counter += 1
        if loss_early_stop_counter >= loss_patience:
            print(f"Early stopping at epoch {epoch+1}: Loss has not improved significantly for {loss_patience} epochs.")
            break

# 确保 model_cnn 目录存在
model_dir = 'model_cnn'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 保存模型（以图标名称命名）
model_save_path = os.path.join(model_dir, f"{icon_name}_classifier.pth")
torch.save(model.state_dict(), model_save_path)
print(f"模型已保存到: {model_save_path}")