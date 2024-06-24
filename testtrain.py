import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load
from glob import glob
import os
import sys  # 导入sys模块，用于接收命令行参数

def calculate_histogram(image_path):
    """计算图像的归一化彩色直方图作为特征向量"""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.calcHist([image], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def predict_image_status(image_path, clf, scaler):
    """使用加载的模型和标准化器来预测图像的状态"""
    hist = calculate_histogram(image_path)
    hist_scaled = scaler.transform([hist])
    prediction = clf.predict(hist_scaled)[0]
    return prediction

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python testsudy.py <icon_name>")
        sys.exit(1)
    
    icon_name = sys.argv[1]  # 从命令行参数获取图标名称

    # 构建模型和标准化器的文件名
    model_filename = f'model/trained_model_{icon_name}.joblib'
    scaler_filename = f'model/scaler_{icon_name}.joblib'

    # 加载模型和标准化器
    clf = load(model_filename)
    scaler = load(scaler_filename)

    # 确定要测试的文件夹
    folders = {}
    for status in ['0', '1']:
        folder_path = f'traindata/{icon_name}-{status}'
        if os.path.exists(folder_path):
            folders[folder_path] = int(status)

    for folder, expected_label in folders.items():
        image_paths = glob(os.path.join(folder, '*.png'))
        correct_predictions = 0
        for image_path in image_paths:
            prediction = predict_image_status(image_path, clf, scaler)
            if prediction == expected_label:
                correct_predictions += 1
            print(f"Image: {image_path} - Expected: {expected_label}, Predicted: {prediction}")
        print(f"Accuracy in {folder}: {correct_predictions / len(image_paths) * 100:.2f}%")
