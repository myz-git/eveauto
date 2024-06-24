import cv2
import numpy as np
import pyautogui
import time
import os
from joblib import load

def scollscreen():
    """水平转动屏幕"""
    fx, fy = pyautogui.size()
    pyautogui.moveTo(400, 50)
    pyautogui.dragRel(-50, 0, 1, pyautogui.easeOutQuad)

def capture_screen_area(region):
    """捕获屏幕上指定区域的截图并转换为OpenCV格式"""
    screenshot = pyautogui.screenshot(region=region)
    screen_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screen_image


def predict_icon_status(image, clf, scaler):
    """通过机器学习的模型来验证图标状态"""
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 转换到HSV颜色空间
    hist = cv2.calcHist([image_hsv], [0, 1, 2], None, [16, 16, 16], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist_scaled = scaler.transform([hist.flatten()])
    prediction = clf.predict(hist_scaled)[0]
    return prediction == 1

def find_icon(template, width, height, clf, scaler, max_attempts=10, offset_x=0, offset_y=0, region=None, exflg=False):
    if region is None:
        fx, fy = pyautogui.size()
        region = (0, 0, fx, fy)

    attempts = 0
    """
    while attempts < max_attempts:
        screen = capture_screen_area(region)
        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        gray_screen = cv2.GaussianBlur(gray_screen, (5, 5), 0)  # 应用高斯模糊
        res = cv2.matchTemplate(gray_screen, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    """
    while attempts < max_attempts:
        screen = capture_screen_area(region)
        # 确保模板和搜索图像都是彩色或都是灰度
        # 如果你的模板是彩色的，确保不要转换成灰度
        # 如果你的模板是灰度的，确保图像也是灰度
        # res = cv2.matchTemplate(cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY), template, cv2.TM_CCOEFF_NORMED)
        res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)  # 使用彩色图像进行模板匹配
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)


        print(f"Attempt {attempts + 1}, max_val: {max_val}")  # 打印当前的最大匹配值

        if max_val > 0.8:
            icon_image = screen[max_loc[1]:max_loc[1] + height, max_loc[0]:max_loc[0] + width]
            if predict_icon_status(icon_image, clf, scaler):
                x = max_loc[0] + region[0] + width // 2 + offset_x
                y = max_loc[1] + region[1] + height // 2 + offset_y
                pyautogui.moveTo(x, y)
                pyautogui.click()
                return True
        attempts += 1
        #scollscreen()
        time.sleep(0.5)

    if exflg:
        print("Failed to find icon after maximum attempts. Exiting...")
        exit(1)
    return False


def main():
    """加载模型和标准化器"""
    # 小黄门模型
    clf_jump0 = load('model/trained_model_jump0.joblib')
    scaler_jump0 = load('model/scaler_jump0.joblib')
    icon_path_jump0 = os.path.join('icon', 'jump0-1.png')
    #template_jump0 = cv2.imread(icon_path_jump0, cv2.IMREAD_COLOR)
    #template_gray_jump0 = cv2.cvtColor(template_jump0, cv2.COLOR_BGR2GRAY)
    template_jump0 = cv2.imread(icon_path_jump0)
    w_jump0, h_jump0 = template_jump0.shape[1], template_jump0.shape[0]  # 修改宽高的获取方式

    
    """ 设置需要捕获的屏幕区域 """
    fx,fy=pyautogui.size()
    
    #右侧面板(全)
    x0, y0, width0, height0 = 1380, 30, 540, 1000
    region0=(x0,y0,width0,height0)



    """Start"""
    time.sleep(1)  # 等待开始

    #pyautogui.moveTo(450,50)
    pyautogui.scroll(200)


    # 持续查找小黄门
    print('查找[小黄门]...')
    while True : 
        if find_icon(template_jump0, w_jump0, h_jump0, clf_jump0, scaler_jump0,1,0,0,region0):
            print('找到[小黄门]!')
            pyautogui.rightClick()         
            break
        # 尝试总览往下划动
        pyautogui.moveTo(1600,400)
        pyautogui.scroll(-900)
        time.sleep(1)
        print('未找到[小黄门],再次查找...')


if __name__ == "__main__":
    main()


