import numpy as np
import pyautogui
import time
from joblib import load
import pynput

# 内部程序调用
from say import speak
from close import close_icons_main
from utils import scollscreen, capture_screen_area, predict_icon_status, load_model_and_scaler,find_icon,load_config
from model_config import models, templates, screen_regions
from speech import speak


def main():
    """加载模型和标准化器"""
    clf_jump0, scaler_jump0 = models['jump0']
    template_jump0, w_jump0, h_jump0 = templates['jump0']
    
    clf_jump1, scaler_jump1 = models['jump1']
    template_jump1, w_jump1, h_jump1 = templates['jump1']

    clf_jump2, scaler_jump2 = models['jump2']
    template_jump2, w_jump2, h_jump2 = templates['jump2']

    clf_jump3, scaler_jump3 = models['jump3']
    template_jump3, w_jump3, h_jump3 = templates['jump3']

    clf_out1, scaler_out1 = models['out1']
    template_out1, w_out1, h_out1 = templates['out1']

    clf_zhongdian2, scaler_zhongdian2 = models['zhongdian2']
    template_zhongdian2, w_zhongdian2, h_zhongdian2 = templates['zhongdian2']

    clf_tingkao, scaler_tingkao = models['tingkao']
    template_tingkao, w_tingkao, h_tingkao = templates['tingkao']

    # 屏幕区域配置
    region_full_right = screen_regions['full_right_panel']
    region_upper_right = screen_regions['upper_right_panel']
    mid_left_panel = screen_regions['mid_left_panel']

    ctr = pynput.keyboard.Controller()
    """Start"""
    speak("开始运行")
    time.sleep(1)  # 等待开始
    close_icons_main()
    #pyautogui.hotkey('ctrl', 'w')
    with ctr.pressed(pynput.keyboard.Key.ctrl,'w'):
        time.sleep(0.3)
        pass                             
    time.sleep(0.5)    
    ## pyautogui.moveTo(450, 50)
    ## pyautogui.scroll(200)

    
    # 查找是否是[停靠]  (目的地在星系内,无需跳跃,直接停靠)
    if find_icon(template_tingkao, w_tingkao, h_tingkao, clf_tingkao, scaler_tingkao, 3, 0, 0, mid_left_panel):
            pyautogui.leftClick()
            print('准备停靠空间站')
            speak('准备停靠空间站')
            # 直接停靠空间站
            while True:
                if find_icon(template_out1, w_out1, h_out1, clf_out1, scaler_out1,1,0,0,region_full_right):
                    speak('已到达目的地')
                    break
                time.sleep(1)
            # 结束主函数，不再执行后续代码
            speak('本次航程结束')
            return


    # 查找是否有[设置为终点]
    if find_icon(template_zhongdian2, w_zhongdian2, h_zhongdian2, clf_zhongdian2, scaler_zhongdian2, 3, 0, 0, mid_left_panel):
            pyautogui.leftClick()
            print('已设置终点!')
            speak('已设置终点!')

    # 持续查找小黄门
    print('查找[小黄门]...')
    while True : 
        if find_icon(template_jump0, w_jump0, h_jump0, clf_jump0, scaler_jump0,1,0,0,region_full_right):
            print('找到[小黄门]!')
            pyautogui.leftClick()            
            break
        # 尝试总览往下划动
        pyautogui.moveTo(1600,400)
        pyautogui.scroll(-900)
        print('未找到[小黄门],再次查找...')
    
    with ctr.pressed('v'):
        time.sleep(0.3)
        pass
    # 第一次跳跃
    print('第一次跳跃...')
    speak('开始航行,启动跃迁引擎')
    if find_icon(template_jump1, w_jump1, h_jump1, clf_jump1, scaler_jump1,3,0,0,region_full_right):
        pyautogui.leftClick()
        print('开始第一次跳跃...')

    time.sleep(3)

    print('第一次跃迁...')
    if find_icon(template_jump2, w_jump2, h_jump2, clf_jump2, scaler_jump2,3,0,0,region_full_right):
        pyautogui.leftClick()
        print('开始第一次跃迁...')
        
    #peak("欢迎登机, 您所乘坐的航班即将起飞,请收起小桌板,调整座椅靠背,手机设置飞行模式")

    """持续航行中"""
    while True:
        with ctr.pressed('v'):
            time.sleep(0.3)
            pass
        time.sleep(1)
        # 持续检查是否到达目的地(可以停靠空间站)
        if find_icon(template_jump3, w_jump3, h_jump3, clf_jump3, scaler_jump3,2,0,0,region_full_right):
            pyautogui.leftClick()            
            print('发现目的地空间站,准备停靠..')
            speak('发现目的地空间站,准备停靠..')
            time.sleep(1)
            # 只有停靠空间站才能退出循环
            break
        else:
            # 检查是否跳跃图标状态
            find_icon(template_jump1, w_jump1, h_jump1, clf_jump1, scaler_jump1,1,0,0,region_full_right)
            pyautogui.leftClick()
            time.sleep(1)
            find_icon(template_jump2, w_jump2, h_jump2, clf_jump2, scaler_jump2,1,0,0,region_full_right)
            time.sleep(1)
            pyautogui.leftClick()
            #speak('正在跃迁中')

            '''
            if find_icon(template_jump1, w_jump1, h_jump1, clf_jump1, scaler_jump1,1,0,0,region_full_right):
                pyautogui.leftClick() 
                print('跳跃至星门')
                speak('跳跃中')
            else:
                # 如果跳跃不可用,则检查跃迁
                if find_icon(template_jump2, w_jump2, h_jump2, clf_jump2, scaler_jump2,1,0,0,region_full_right):
                    pyautogui.leftClick()
                    print('开始跃迁')
                    speak('跃迁中')
            '''
        # 重复检查
        time.sleep(2)

    # 停靠空间站
    while True:
        print('正在停靠空间站')
        if find_icon(template_out1, w_out1, h_out1, clf_out1, scaler_out1,3,0,0,region_full_right):
            #speak('您的旅程已结束,感谢乘坐天合联盟东方航空,祝您旅途愉快',180)
            print('空间站已停靠!')
            speak('空间站已停靠!')
            break
        time.sleep(1)
    speak('本次航程结束')

if __name__ == "__main__":
    main()


