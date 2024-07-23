import cv2
import os
from joblib import load

def load_model_and_scaler(base_path, model_name):
    clf = load(os.path.join(base_path, f'trained_model_{model_name}.joblib'))
    scaler = load(os.path.join(base_path, f'scaler_{model_name}.joblib'))
    return clf, scaler

def load_template(icon_path):
    template = cv2.imread(icon_path, cv2.IMREAD_COLOR)
    return template, template.shape[1], template.shape[0]  # width, height

# 定义图标模板和模型路径
base_path = 'model'
icon_base_path = 'icon'

# 模型和模板加载
"""
jump0 小黄门 
jump1 跳跃模型
jump2 跃迁
jump3 停靠
zhongdian2 终点
out1 离站
close1 关闭窗口
agent1 代理人
agent2 代理人-我要执行新任务
agent3 代理人-接受
yunshumubiao1 运输目标
search2 仓库搜索
jiku11 机库 
jiku2 仓库
chakanrenwu1 查看任务
kjz1 空间站(1)
zhongdian1 设置为终点
search1  "地点 搜素"
zonglan1 总览
"""
model_names = ['jump0', 'jump1', 'jump2', 'jump3', 'zhongdian2', 'out1','close1','agent1','agent2','agent3','yunshumubiao1','search2','jiku1','jiku2','chakanrenwu1','talk1','talk2','kjz1','search1','zhongdian1','zonglan1']
models = {name: load_model_and_scaler(base_path, name) for name in model_names}
templates = {name: load_template(os.path.join(icon_base_path, f'{name}-1.png')) for name in model_names}

# 屏幕区域配置
screen_regions = {
    'full_right_panel': (1380, 30, 540, 1000), #右侧面板(全)
    'upper_right_panel': (1380, 30, 540, 260), #右侧面板(上)
    'mid_left_panel': (50, 150, 500, 600), #左侧面板(左中)
    'agent_panel1': (1450, 250, 500, 500), #代理人
    'agent_panel2': (1500, 400, 400, 500), #代理人列表
    #'agent_panel3': (400, 100, 1200, 650), #代理人对话窗口
    'agent_panel3': (200, 100, 1400, 900), #代理人对话窗口
    'cangku_panel3': (0, 0, 1700, 850), #仓库
    'need_goods_panel':(50,50,400,500) #任务目标提示
}
