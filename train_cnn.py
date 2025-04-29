import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os

# 定义改进后的 CNN 模型
class IconCNN(nn.Module):
    def __init__(self):
        super(IconCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 3, 256)
        self.fc2 = nn.Linear(256, 2)  # 2 类：jump1-1、jump1-0
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 3)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((32, 24)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载数据集
dataset = datasets.ImageFolder('traindata', transform=transform)

# 打印所有类别
print("所有类别:", dataset.classes)

# 确保只使用 jump1-0 和 jump1-1
desired_classes = ['jump1-0', 'jump1-1']
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
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 训练循环
num_epochs = 30
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
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

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
    print(f"Validation Accuracy: {100 * correct / total}%")

# 保存模型
torch.save(model.state_dict(), 'icon_classifier.pth')