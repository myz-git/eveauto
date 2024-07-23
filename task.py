import numpy as np
import pyautogui
import pyperclip  # 导入 pyperclip
import time
from joblib import load
from cnocr import CnOcr
import re

# 内部程序调用
from say import speak
from close import close_icons_main
from outsite import outsite_icons_main,outsite_check
from utils import scollscreen, capture_screen_area, predict_icon_status, load_model_and_scaler,find_icon,load_location_name,find_txt_ocr,find_txt_ocr2,correct_string
from model_config import models, templates, screen_regions

class IconNotFoundException(Exception):
    """Exception raised when an icon is not found."""
    pass

class GoodsNotFoundException(Exception):
    """Exception raised when the specified goods are not found."""
    pass

class TextNotFoundException(Exception):
    """Exception raised when the specified text is not found."""
    pass



def extract_goods_name(text):
    """从文本中提取指定格式的商品名称"""
    pattern = re.compile(r'货物\d*×(.+?)\*')
    match = pattern.search(text)
    if match:
        goods_name = match.group(1).strip()  # 提取并去除可能的前后空白
        return goods_name
    return None  # 如果没有匹配到，返回 None


def get_goods(template, width, height, clf, scaler, max_attempts=3, offset_x=0, offset_y=0, region=None, exflg=False):
        # 定位运输目标
        # "获得任务目标,找不到则程序终止"
        goods=None
        if find_icon(template, width, height, clf, scaler,max_attempts,offset_x,offset_y,region,exflg):
            print("--查找运输目标")        
            time.sleep(0.5)
            # 获取货物内容, 根据运输目标坐标直接定位货物右侧的OCR扫描范围
            x, y = pyautogui.position()         
            region_hw=(x+80,y+94,280,50)
            print(f"--region_hw={region_hw}")
            
            # 只匹配汉字，如果折行只提取第一行           
            goods = find_txt_ocr2('', 3, region_hw)
            return goods
            #except Exception as e:
            #    print(f"未取得运输目标，错误信息: {str(e)}")
            #    raise GoodsNotFoundException("未取得运输目标")


def move_goods(goods):
    # 从仓库搬运货物到舰船机库
    # 打开仓库
    print("3.1 打开仓库")
    pyautogui.hotkey('alt', 'c')
    time.sleep(0.5)
    cangku_panel3 = screen_regions['cangku_panel3']

    clf_search2, scaler_search2 = models['search2']
    template_search2, w_search2, h_search2 = templates['search2']

    clf_jiku1, scaler_jiku1 = models['jiku1']
    template_jiku1, w_jiku1, h_jiku1 = templates['jiku1']

    clf_jiku2, scaler_jiku2 = models['jiku2']
    template_jiku2, w_jiku2, h_jiku2 = templates['jiku2']
    # 获取机库坐标   
    print("3.2 找机库坐标")
    if find_icon(template_jiku1, w_jiku1, h_jiku1, clf_jiku1, scaler_jiku1,2,20,0,cangku_panel3):
        print("3.2.1 找机库坐标")
        jiku_x,jiku_y = pyautogui.position() 

        # 激活仓库   
        if find_icon(template_jiku2, w_jiku2, h_jiku2, clf_jiku2, scaler_jiku2,2,0,0,cangku_panel3):
            print("3.2.2 激活仓库")
            pyautogui.leftClick()
            time.sleep(0.2)

        # 8.搜索仓库
        print("3.3 查找仓库搜索栏")
        if find_icon(template_search2, w_search2, h_search2, clf_search2, scaler_search2,5,0,0,cangku_panel3):
            print("3.4 搜索仓库..")
            pyautogui.leftClick()        
            #pyperclip.copy(goods)  # 复制名称到剪贴板
            pyperclip.copy(goods[:4])  # 复制截取前4个汉字到剪贴板
            #pyperclip.copy(goods[2:])  # 复制截取从第2位开始取3个汉字到剪贴板  (避免识别 "一大群.."为"一天群")
            pyautogui.leftClick() 
            time.sleep(0.5)
            pyautogui.hotkey('ctrl', 'v')  # 粘贴名称
            pyautogui.press('enter')
            time.sleep(1)
            pyautogui.moveRel(-50,65)
            time.sleep(0.5)
            pyautogui.dragTo(jiku_x+10,jiku_y,1,pyautogui.easeOutQuad)
            time.sleep(0.5)
            pyautogui.hotkey('ctrl', 'w')
            print(f"3.5 [{goods}]已放入机库中,请检查...")

def check_goods(cangku_panel3):
    print("通过OCR文字识别是否有[货柜缺失]弹窗")
    try:
        # 尝试找到 "货柜缺失" 文本
        if find_txt_ocr('货柜缺失', 1, cangku_panel3):
            pyautogui.leftClick()
            time.sleep(0.2)
            pyautogui.hotkey('ctrl', 'w')
            return False
    except TextNotFoundException:
        # 如果没有找到，说明没有错误弹窗，货物正常
        return True

    # 默认返回 True，表示没有问题
    return True


        

def main():
    """加载模型和标准化器"""
    #代理人列表窗口
    clf_agent1, scaler_agent1 = models['agent1']
    template_agent1, w_agent1, h_agent1 = templates['agent1']
    
    clf_agent2, scaler_agent2 = models['agent2']
    template_agent2, w_agent2, h_agent2 = templates['agent2']

    clf_agent3, scaler_agent3 = models['agent3']
    template_agent3, w_agent3, h_agent3 = templates['agent3']

    clf_yunshumubiao1, scaler_yunshumubiao1 = models['yunshumubiao1']
    template_yunshumubiao1, w_yunshumubiao1, h_yunshumubiao1 = templates['yunshumubiao1']

    clf_out1, scaler_out1 = models['out1']
    template_out1, w_out1, h_out1 = templates['out1']

    clf_zonglan1, scaler_zonglan1 = models['zonglan1']
    template_zonglan1, w_zonglan1, h_zonglan1 = templates['zonglan1']
    

    clf_chakanrenwu1, scaler_chakanrenwu1 = models['chakanrenwu1']
    template_chakanrenwu1, w_chakanrenwu1, h_chakanrenwu1 = templates['chakanrenwu1']

    


    #设置需要捕获的屏幕区域
    agent_panel1 = screen_regions['agent_panel1']
    #代理人列表窗口
    agent_panel2 = screen_regions['agent_panel2']
    #代理人对话窗口
    agent_panel3 = screen_regions['agent_panel3']
    #货物缺失提示窗口
    need_goods_panel = screen_regions['need_goods_panel']
    cangku_panel3 = screen_regions['cangku_panel3']


    # 01. 准备开始
    print("task start") 
    time.sleep(1)
    print("0 task sleep(1)") 

    # 02. 查找"代理人"图标  
    print("0.2.0 查找[代理人]图标") 
    try:  
        if find_icon(template_agent1, w_agent1, h_agent1, clf_agent1, scaler_agent1,2,0,0,agent_panel1):
            pyautogui.leftClick()
    except IconNotFoundException as e:
        print(e)

    # 03. 查找代理人
    # 0.3.1 获得代理人名字
    print("0.3.0 获得代理人名字") 
    agent_name = load_location_name('agent')
    print(f"0.3.1 agent={agent_name}")

    # 0.3.2 通过OCR文字识别查找代理人
    print(f"0.3.2 通过OCR文字识别查找代理人")
    if find_txt_ocr(agent_name,1,agent_panel2):
        pyautogui.hotkey('ctrl', 'w')
        pyautogui.doubleClick()  # 双击打开代理人对话窗口
        time.sleep(0.5)

    # 0.4. 和代理人开始对话

    """↓↓↓ 低安特供 ↓↓↓"""
    #"""
    try:
        goods=get_goods(template_yunshumubiao1, w_yunshumubiao1, h_yunshumubiao1, clf_yunshumubiao1, scaler_yunshumubiao1,2,0,0,agent_panel3)
        print(f"0.5.0 低安任务直接获得货物:{goods}")
        # 查找[接受]任务按钮
        if find_icon(template_agent3, w_agent3, h_agent3, clf_agent3, scaler_agent3,2,0,0,None):
            pyautogui.leftClick()
            print("0.5.1 低安已接受任务!")
            pyautogui.hotkey('ctrl', 'w')
            time.sleep(0.5)
    except GoodsNotFoundException as e:
        print(e)
   #"""
    """↑↑↑ 低安特供 ↑↑↑"""


    # 当还未提供任务
    print("1.0 正式开始读取任务")
    if find_icon(template_agent2, w_agent2, h_agent2, clf_agent2, scaler_agent2,2):
            print("1.1 我要执行新任务")
            pyautogui.leftClick()
            time.sleep(0.5)
            #...get goods
            try:
                goods=get_goods(template_yunshumubiao1, w_yunshumubiao1, h_yunshumubiao1, clf_yunshumubiao1, scaler_yunshumubiao1,2,0,0,agent_panel3)
                print(f"1.2 获得货物:{goods}")
            except GoodsNotFoundException as e:
                print(e)
            
            # 查找[接受]任务按钮
            if find_icon(template_agent3, w_agent3, h_agent3, clf_agent3, scaler_agent3,2,0,0,None):
                pyautogui.leftClick()
                print("1.3 已接受任务!")
                pyautogui.hotkey('ctrl', 'w')
                time.sleep(0.5)

    # 查看任务
    else:
        # 当有查看任务时
        if find_icon(template_chakanrenwu1, w_chakanrenwu1, h_chakanrenwu1, clf_chakanrenwu1, scaler_chakanrenwu1,2):
            print("2.1 查看任务")
            pyautogui.leftClick()
            time.sleep(0.5)
            try:
                goods=get_goods(template_yunshumubiao1, w_yunshumubiao1, h_yunshumubiao1, clf_yunshumubiao1, scaler_yunshumubiao1,3,0,0,agent_panel3)
                print(f"2.2 获得货物:{goods}")
            except GoodsNotFoundException as e:
                print(e)
            
            # 查找[接受]任务按钮
            if find_icon(template_agent3, w_agent3, h_agent3, clf_agent3, scaler_agent3,1,0,0,None):
                pyautogui.leftClick()
                print("2.3 已接受任务!")
                pyautogui.hotkey('ctrl', 'w')
                time.sleep(0.5)

            # 没有[接受]任务按钮
            else:
                print("2.4 之前接受过任务!")
                pyautogui.hotkey('ctrl', 'w')
                time.sleep(0.5)

    #pyautogui.hotkey('ctrl', 'w')
    
    #对OCR识别结果进行修正
    goods=correct_string(goods)
    print(f"任务目标:运送[{goods}]")  
    
    # 从仓库移动货物到船仓
    #subgood=goods[:-1] #截取货物名称前三位
    #move_goods(subgood)
    time.sleep(0.5)
    print(f"准备搬运{goods}")
    pyautogui.hotkey('alt', 'f')
    print(f"准备搬运2222{goods}")
    time.sleep(5)
    move_goods(goods)
                                                                   
    # 9. 出站
    close_icons_main()    
    pyautogui.hotkey('ctrl', 'w')
    
    """
    # 出站, 并检查出站时是否货柜缺失弹窗
    time.sleep(1)
    attempts = 0
    max_attempts = 10
    while attempts < max_attempts:
        try:
            if find_icon(template_out1, w_out1, h_out1, clf_out1, scaler_out1, 3):
                pyautogui.leftClick()
                time.sleep(0.5)

            if check_goods(cangku_panel3):
                break  # 如果货柜正常，则退出循环
            
            goodstmp = find_txt_ocr2('你需要有', 3, need_goods_panel)
            if goodstmp:
                #pyperclip.copy(goodstmp[-5:-2])  # 获取并处理货物名称
                pyperclip.copy(goodstmp[:3])  # 获取并处理货物名称(前三位)
                move_goods(goodstmp)

            time.sleep(0.5)
            scollscreen()
            attempts += 1
        except (IconNotFoundException, TextNotFoundException) as e:
            print(e)
            time.sleep(0.5)  # 出现异常时等待一段时间再重试

    """
    print("出站中,请等待...")
    outsite_icons_main()
    outsite_check()

if __name__ == "__main__":
    main()
