import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump
from glob import glob
import os
import sys  # 导入sys模块，用于接收命令行参数

def calculate_histogram(image_path):
    """计算给定图像路径的彩色直方图并进行归一化处理，作为特征向量"""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Warning: Unable to load image {image_path}")
        raise ValueError(f"Unable to load image: {image_path}")
    else:
        print(f"Loaded image: {image_path}")
    #hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.calcHist([image], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def load_images_from_folder(folder):
    """从指定文件夹加载所有图片的路径，并打印出加载的图片数量"""
    image_paths = glob(os.path.join(folder, '*.png'))
    print(f"Found {len(image_paths)} images in {folder}")
    return image_paths

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python study.py <icon_name>")
        sys.exit(1)

    icon_name = sys.argv[1]  # 从命令行参数获取图标名称

    # 构建路径
    available_icon_folder = f'traindata/{icon_name}-1'
    unavailable_icon_folder = f'traindata/{icon_name}-0'

    available_icon_paths = load_images_from_folder(available_icon_folder)
    unavailable_icon_paths = load_images_from_folder(unavailable_icon_folder)
    icon_paths = available_icon_paths + unavailable_icon_paths
    labels = [1] * len(available_icon_paths) + [0] * len(unavailable_icon_paths)

    features = []
    for path in icon_paths:
        try:
            hist = calculate_histogram(path)
            features.append(hist)
        except ValueError as e:
            print(e)

    features = np.array(features)
    labels = np.array(labels)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 初始化随机森林模型并进行训练
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # 评估模型在测试集上的性能
    y_pred = clf.predict(X_test_scaled)
    print("测试集上的准确率:", accuracy_score(y_test, y_pred))

    # 保存训练好的模型和标准化器
    model_filename = f'model/trained_model_{icon_name}.joblib'
    scaler_filename = f'model/scaler_{icon_name}.joblib'
    dump(clf, model_filename)
    dump(scaler, scaler_filename)
    print(f"Model and scaler saved: {model_filename}, {scaler_filename}")
