import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from glob import glob

def calculate_features(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        # 计算彩色直方图
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist = hist.flatten()
        # 转换为灰度并计算LBP
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray_image, P=24, R=3, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=np.arange(27), range=(0, 26))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)  # 归一化
        # 合并特征
        combined_features = np.concatenate((hist, lbp_hist))
        return combined_features
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# 收集数据和标签
icon_paths = glob('jump1-1-f*.png') + glob('jump1-0-f*.png')
labels = [1] * 5 + [0] * 5  # 前5个为可用, 后5个为不可用

# 计算特征
features = []
new_labels = []

for path, label in zip(icon_paths, labels):
    feature = calculate_features(path)
    if feature is not None:
        features.append(feature)
        new_labels.append(label)

features = np.array(features)
new_labels = np.array(new_labels)

# 检查特征和标签的长度
print(f"Features length: {len(features)}, Labels length: {len(new_labels)}")

# 拆分训练和测试数据集
X_train, X_test, y_train, y_test = train_test_split(features, new_labels, test_size=0.2, random_state=42)

# 特征归一化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# 验证分类器性能
y_pred = clf.predict(X_test_scaled)
print("测试集上的准确率:", accuracy_score(y_test, y_pred))


# 预测新图像状态的函数
def predict_icon_status(image_path, clf, scaler):
    features = calculate_features(image_path)
    features_scaled = scaler.transform([features])
    pred = clf.predict(features_scaled)[0]
    return "可用" if pred == 1 else "不可用"

# 使用实际图像测试
test_image_path = 'jump1-0-f3.png'
status = predict_icon_status(test_image_path, clf, scaler)
print("新图标状态:", status)
