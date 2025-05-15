## 项目描述

实现游戏中自动执行物流任务,分为如下步骤:
a. 接任务: 在空间站与代理人谈话接任务,然后将任务物品从仓库挪到飞船仓中(task.py );
b. 送货: 自动驾驶到目的地空间站(jump1.py);
c. 交任务:  代理人谈话提交任务,然后设定返程导航(talk.py,navigate.py);
d. 返程: 自动驾驶返回原来空间站(jump1.py);
如上步骤自动循环执行;



## 项目架构

|------eveauto------
|-go.py 
| 功能:启动及调度程序
| 说明:实现按顺序循环执行"接任务,送货,交任务(设定返程导航),返回"四个步骤
|-
|-task.py
| 功能:接任务程序;
| 说明:在空间站与代理人谈话接任务,然后将任务物品从仓库挪到飞船仓中,离开空间站;
|
|-jump1.py
| 功能:送货;
| 说明: 设定送货任务目的地,自动驾驶到目的地空间站;
|
|-talk.py
| 功能:交任务;
| 说明:在目的地空间站和代理人谈话提交任务;
|
|-navigate.py
| 功能:回程导航
|说明:设定返回目的地的导航,离开空间站;
|
|-outsite.py
| 功能:离开空间站
| 说明: 让飞船离开当前空间站;
|
|-say.py
| 功能: 语音播报
| 说明:语音播报;
|
|-utils.py
| 功能:通用功能函数
| 说明:包括图像识别,文字识别等通用函数
|
|-jump2.py
| 功能: 自动驾驶程序
| 说明: 其他场景中的自动驾驶程序
|
|-----------------------

|----模型训练相关-------
|-snap.py
|-功能:自动抓取图标
|-说明:根据给定的icon/xxx.png样例图标,自动抓取不同光线背景下的图标并保存在训练文件夹(traindata);
|-
|-train.py
|-功能: CNN训练模型
|-说明:使用traindata素材生成CNN模型(在model_cnn下),能够识别不同背景变化（如黑色、灰白色、灰色背景）
|-
|-verify.py
|-功能:模型验证 
|-说明: 对训练的模型进行反向验证 ,查看验证反馈(含外形匹配验证及CNN识别验证)
|-
|-icon  需要识别的图标模板
|-traindata 用来训练的图标素材
|-model_cnn训练后的图标模型
|-----------------------





## 环境准备

 **环境**：使用 Python 3.10.6，PyTorch 2.3.1，OpenCV 4.9.0.80，PyAutoGUI 0.9.54，屏幕分辨率为 1920x1080

```
cd D:\Workspace\git\eveauto
conda remove -n evejump --all
conda create -n evejump python=3.10.6
conda activate evejump
##pip install -r requirements.txt
##pip uninstall torch
##pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
##pip install scipy
##pip install pypinyin

pip install pillow pyautogui 
pip install opencv-python numpy
pip install scikit-learn scikit-image
pip install 

pip install cnocr
pip install onnxruntime
##pip uninstall cnocr[ort-gpu]
##pip install cnocr[ort-gpu]
##pip uninstall onnxruntime
##pip install onnxruntime-gpu

pip install pyttsx3
pip install tensorflow
pip install keyboard
pip install pynput

```

## 关键技术

1. **图像捕获与处理**：
   
   - 使用 `pyautogui` 和 `OpenCV` 捕获指定屏幕区域的图像。
   - 处理图像以便后续的模板匹配和状态识别。
   
2. **模板匹配**：
   
   - 利用 `OpenCV` 的模板匹配功能来定位屏幕上的图标。使用了图标的灰度模板与屏幕截图的灰度版本进行匹配。
   
   - 计算图标的精确位置，包括左上角和中心点坐标。
   
3. **图像识别**：
   
   - 通过模板匹配外形(icon/*.png)第一步筛选;
   - 使用训练好的cnn模型来对图像在不同光线背景下的识别;
   
4. **文字识别**：
   
   使用CNOCR
   
4. **自动操作执行**：
   - 使用pynput.keyboard及pyautogui 控制键鼠
   
   

## 遇到的问题与解决方案：

- **图标位置识别不准确**：
  通过调整模板匹配的参数和确保坐标系统一致性（将相对坐标转换为绝对屏幕坐标）解决了问题。
- 



## 图像识别步骤

### 制作图标模板

截取游戏界面清晰背景的下标准图标作为模板,保存在icon下, 如  icon/jump1-1.png;

 注:  图标背景要清晰纯色 ,能够清晰表现图标外形



### 采集图标素材

执行 python snap.py jump1-1.png  , 然后缓慢移动游戏界面,表现出图标不同的光线背景的展现;  此时snap.py会对该图标进行持续截图并保存在traindata/jump1-1/下作为训练素材

可重复多次该步骤,尽量覆盖各种光线背景



### 使用图标素材训练模型

执行python  train.py  jump1-1 对图标(如jump1-1.png, jump3-0.png)  进行学习训练得到模型model_cnn/jump1-1_classifier.pth



### 模型验证 

对训练模型进行反向验证 ,查看验证反馈, 

执行python verify.py  jump1-1  验证游戏界面中是否可以正确识别到图标, 没有漏检和误检, 可能需要调整外形匹配阈值及CNN识别阈值;

### 加载和使用模型

utils.py 中 find_icon_cnn函数

```

def find_icon_cnn(icon, region, max_attempts=3, offset_x=0, offset_y=0, threshold=0.86):
    if region is None:
      fx, fy = pyautogui.size()
      region = (0, 0, fx, fy)    
    
    """使用模板匹配和CNN模型在屏幕特定区域查找图标，并将鼠标移动到图标中心"""
    # 加载模板图标
    template_path = os.path.join('icon', f"{icon}.png")
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"模板图像 '{template_path}' 无法加载")
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    icon_height, icon_width = template.shape[:2]
    w, h = template_gray.shape[::-1]

    # 定义 CNN 模型
    class IconCNN(nn.Module):
        def __init__(self):
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
            self.dropout = nn.Dropout(0.5)
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
    model = IconCNN().to(device)
    model_path = os.path.join('model_cnn', f"{icon}_classifier.pth")
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please ensure the model file exists.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("The model may have a different structure. Please retrain using 'train.py'.")
        sys.exit(1)
    model.eval()

    # 数据预处理（CNN 输入）
    transform = transforms.Compose([
        transforms.Resize((icon_height, icon_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 动态确定类别
    opposite_icon = f"{icon.split('-')[0]}-0"
    if os.path.exists(os.path.join('traindata', opposite_icon)):
        class_names = [icon, opposite_icon]
    else:
        class_names = [icon, 'other']

    attempts = 0
    while attempts < max_attempts:
        screen_image = capture_screen_area(region)
        screen_gray = cv2.cvtColor(screen_image, cv2.COLOR_BGR2GRAY)

        # 模板匹配
        res = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        matches = list(zip(*loc[::-1]))

        if matches:
            for pt in matches:
                top_left = pt
                bottom_right = (top_left[0] + w, top_left[1] + h)
                icon_image = screen_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                icon_rgb = cv2.cvtColor(icon_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(icon_rgb)
                tensor = transform(pil_image).unsqueeze(0).to(device)

                # 使用 CNN 模型推理
                with torch.no_grad():
                    output = model(tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    predicted_class = class_names[predicted.item()]
                    print(f"坐标 ({top_left[0] + region[0]}, {top_left[1] + region[1]}) 模板匹配度: {res[top_left[1], top_left[0]]:.4f}, 预测类别: {predicted_class}, 置信度: {confidence.item():.4f}")

                    if predicted.item() == 0 and confidence.item() > 0.85:
                        x = top_left[0] + region[0] + w // 2 + offset_x
                        y = top_left[1] + region[1] + h // 2 + offset_y
                        pyautogui.moveTo(x, y)
                        return True

            print(f"找到 {len(matches)} 个匹配区域，但都不是 {icon}（可能是 {class_names[1]}）")
        else:
            print(f"未找到图标，最高匹配度: {np.max(res):.4f}")

        attempts += 1
        time.sleep(0.5)
        scollscreen()

    return False
```



## FYI

### OCR

http://masikkk.com/article/Tesseract/

cnocr:  https://github.com/breezedeus/cnocr

使用方法:*https://cnocr.readthedocs.io/zh/latest/usage*



#### TASK

https://wiki.eveuniversity.org/Missions

*/open journal*
