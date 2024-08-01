## 项目目的

实现持续自动执行物流任务;



## 项目架构

|------eveauto------
|-go.py 
| 功能:启动及调度程序
| 说明:实现按顺序循环执行"接任务,送货,交任务,回程导航,返回"五个步骤,对应调用'task.py', 'jump1.py', 'talk.py', 'navigate.py', 'jump2.py';
|-
|-task.py
| 功能:接任务程序;
| 说明:实现在空间站自动找代理人谈话并接受任务,获得货物;
|
|-jump1.py
| 功能:送货;
| 说明:设定送货任务目的地,并前往目的地空间站;
|
|-talk.py
| 功能:交任务;
| 说明:在目的地空间站和代理人谈话交任务;
|
|-navigate.py
| 功能:回程导航
|说明:设定返回目的地的导航,并出站;
|
|-jump2.py
| 功能:返回
| 说明:返回接任务的空间站;
|
|-outsite.py
| 功能:出站
| 说明:离开当前空间站;
|
|-say.py
| 功能:播报
| 说明:语音播报;
|
|-utils.py
| 功能:通用功能函数
| 说明:包括图像识别,文字识别,图标状态识别等通用函数
|
|-model_config.py
| 功能:加载模型函数
| 说明:定义图标模板和模型路径,以及图像识别区域配置
|-----------------------

|----模型训练相关-------
|-snap.py
|-功能:自动抓取图标
|-说明:根据给定的样例图标,程序运行时自动抓取不同背景下的图标并保存在训练文件中;
|-
|-train1stat.py
|-功能:训练模型函数(单状态)
|-说明:使用机器学习判断图标状态生成训练模型
|-
|-train2stat.py
|-功能:训练模型函数(双状态)
|-说明:使用机器学习判断图标状态生成训练模型
|-
|-testtrain.py
|-功能:模型验证 
|-说明:对训练模型进行反向验证 ,查看验证反馈
|-
|-icon  需要识别的图标样例
|-traindata 用来训练的图标
|-model 训练后的图标模型
|-----------------------







## 环境准备

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


```

## 实现技术

1. **图像捕获与处理**：
   
   - 使用 `pyautogui` 和 `OpenCV` 捕获指定屏幕区域的图像。
   - 处理图像以便后续的模板匹配和状态识别。
   
2. **模板匹配**：
   - 利用 `OpenCV` 的模板匹配功能来定位屏幕上的图标。使用了图标的灰度模板与屏幕截图的灰度版本进行匹配。
   
   - 计算图标的精确位置，包括左上角和中心点坐标。
   
3. **状态识别**：
   
   - 通过机器学习模型（使用随机森林分类器），根据图像的彩色直方图特征来预测图标的状态（可用或不可用）。
   - 使用 `StandardScaler` 进行特征的标准化处理，确保预测时的数据处理与训练模型时相同。
   
4. **文字识别**：
   
   使用CNOCR
   
4. **自动操作执行**：
   - 根据图标的状态，如果图标可用，则自动移动鼠标到图标的中心坐标并执行点击操作。
   
5. **调试与优化**：
   - 逐步调试程序，确保图标可以准确地被定位和识别。
   - 解决了图像捕获、模板匹配精度、状态预测准确性以及坐标转换等方面的问题。

### 遇到的问题与解决方案：

- **图标位置识别不准确**：通过调整模板匹配的参数和确保坐标系统一致性（将相对坐标转换为绝对屏幕坐标）解决了问题。
- **状态识别总是显示为可用**：确保预测使用的图像预处理与训练时一致，增加了状态预测的调试输出来帮助诊断问题。
- **鼠标操作不准确**：修改了坐标计算方法，确保鼠标准确地移动到图标的中心并进行点击。



## 实现步骤

### 模型准备

1. #### 截取识别图标 

   截取游戏界面清晰背景的可用状态下的图标,保存在icon下, 如  icon/jump3-1.png;

   截取游戏界面清晰背景的不可用状态下的图标,保存在icon下, 如  icon/jump3-0.png;  (如果图标就一种状态,该步骤省略)

    注:  图标保存为33*33大小png格式,背景要清晰纯色 ,能够突显图标;

    如 ,  icon/jump3-1.png(可用状态) , icon/jump3-0.png(不可用状态)  ;

   

2. #### 准备训练图片

   2.1 游戏中在显示jump3-1.png图标时, 执行 python snap.py jump3-1.png  , 然后缓慢移动游戏界面,表现出图标不同的背景的展现;,  此时snap.py会对该图标进行持续人截图并保存到studydata下(默认每秒截取1次,持续100次)  自动保存在studydata/jump3-1下作为训练素

    2.2 重复步骤2.1  在游戏显示不可用状态的图标时,进行截取 python snap.py jump3-0.png   , 自动保存在studydata/jump3-0下;

   

3. #### 使用机器学习判断图标状态生成训练模型

   3.1 执行python  study2sta.py  jump3 对具有两种状态的图标(如jump3-1.png, jump3-0.png)  进行机器学习训练识别能力,获得trained_model_jump3.joblib, scaler_jump3.joblib训练模型;

   3.2  执行python  study1sta.py  jump2对只有一种状态的图标(如jump2-1.png)进行机器学习训练识别能力,获得python  study2sta.py  jump3训练模型

   观察训练结果, 比如:  测试集上的准确率: 0.9696969696969697 

   

4. #### 模型验证 

   对训练模型进行反向验证 ,查看验证反馈, 

   注:teststudy.py可以自动识别一种状态还是两种状态 

   python teststudy.py jump3    

   ...
   Image: studydata/jump3-0\jump3-0-42.png - Expected: 0, Predicted: 0
   Image: studydata/jump3-0\jump3-0-43.png - Expected: 0, Predicted: 1
   Image: studydata/jump3-0\jump3-0-44.png - Expected: 0, Predicted: 0
   ...
   Accuracy in studydata/jump3-0: 98.39%

   表示jump3-0-43.png这个图像不能被正确识别;

   #### 5. 加载和使用模型

   执行最终的识别程序jump.py,调用模型进行识别,并触发相应动作;

   左上角坐标：(1380, 50)
   x1, y1, width1, height1 = 1380, 50, 320, 650 # 设置为需要捕获的屏幕区域

   注意:  确保jump.py与study.py中的特征提取`calculate_histogram` 函数一致 ,如:

   ```
   def calculate_histogram(image):
       """计算图像的归一化彩色直方图作为特征向量"""
       hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
       cv2.normalize(hist, hist)
       return hist.flatten()
   ```

   以及处理图标的方式:

   ```
       # 处理图标1
       res_jump1 = cv2.matchTemplate(panel_gray, template_gray_jump1, cv2.TM_CCOEFF_NORMED)
   ```

   

### 示例: 交任务

#### 1. 准备识别图标 

截图icon/talk1-1.png  大小244*28, 该图标就一种状态;

注:  背景要清晰纯色 ,能够突显图标;

#### 2.准备训练图片

```
python snap.py talk1-1.png
```

多生成几次, 图片保存在studydata\talk1-1下

#### 3. 训练模型

因为只有一种状态,因此使用 study1sta.py

```
python study1sta.py talk1
```

测试集上的准确率: 1.0
Model and scaler saved: trained_model_talk1.joblib, scaler_talk1.joblib

#### 4. 模型验证

```
python teststudy.py talk1
```

...

Image: studydata/talk1-1\talk1-1-98.png - Expected: 1, Predicted: 1
Image: studydata/talk1-1\talk1-1-99.png - Expected: 1, Predicted: 1
Accuracy in studydata/talk1-1: 100.00%

#### 5. 主程序使用模型

##### 5.1 确定捕捉区域

截个游戏界面全图,使用图像软件如GIMP, 确定捕捉区域坐标;

左上角坐标：(0, 0)
x1, y1, width1, height1 = 0, 0, 400, 500 # 设置为需要捕获的屏幕区域







### 示例: 接任务

#### 1. 找代理人谈话

##### 1).准备模型

**a.识别图标agent1** 

​	截图icon/agent1-0.png  ,icon/agent1-1.png  代表两种状态,

​	注:  背景要清晰纯色 ,能够突显图标, 两个截图大小要一样;

**b. 准备0状态(agent1-0)训练图片**

```
python snap.py agent1-0.png
```

​	在游戏中展现agent1-0的界面, 缓慢移动背景,展现不同背景下agent1-0,一两分钟后,停止snap.py, 检查studydata/agent1-0/下抓取的图片是否有不符合的; 

**c. 准备1状态(agent1-1)训练图片**

​	同上;

**d.训练agent1**

```
python study2sta.py agent1
--1个状态用study1sta,2个状态用study2sta
测试集上的准确率: 1.0
Model and scaler saved: trained_model_agent1.joblib, scaler_agent1.joblib
```

得到模型: **trained_model_agent1.joblib, scaler_agent1.joblib**

##### 2). 主程序



## FYI

### OCR

http://masikkk.com/article/Tesseract/

cnocr:  https://github.com/breezedeus/cnocr

使用方法:*https://cnocr.readthedocs.io/zh/latest/usage*



#### TASK

https://wiki.eveuniversity.org/Missions

*/open journal*
