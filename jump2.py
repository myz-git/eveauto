import cv2
import numpy as np
import pyautogui
import time
from joblib import load
import os

def scollscreen(max_attempts=10):
    """转动屏幕"""
    fx,fy=pyautogui.size()
    pyautogui.moveTo(500,200)
    pyautogui.dragRel(-50,-50,0.5,pyautogui.easeOutQuad)
    pyautogui.hotkey('ctrl', 'w')
    pyautogui.scroll(20)

def capture_screen_area(region):
    """捕获屏幕上指定区域的截图并转换为OpenCV格式"""
    screenshot = pyautogui.screenshot(region=region)
    screen_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screen_image

def calculate_histogram(image):
    """计算图像的归一化彩色直方图作为特征向量"""
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def predict_icon_status(image, clf, scaler):
    """使用机器学习模型判断图标状态"""
    hist = calculate_histogram(image)
    hist_scaled = scaler.transform([hist])
    prediction = clf.predict(hist_scaled)[0]
    return prediction == 1  # 返回True如果图标为可用状态，否则False


def main():
    # 加载模型和标准化器
    clf_jump0 = load('model/trained_model_jump0.joblib')
    scaler_jump0 = load('model/scaler_jump0.joblib')
    clf_jump1 = load('model/trained_model_jump1.joblib')
    scaler_jump1 = load('model/scaler_jump1.joblib')
    clf_jump2 = load('model/trained_model_jump2.joblib')
    scaler_jump2 = load('model/scaler_jump2.joblib')
    clf_jump3 = load('model/trained_model_jump3.joblib')
    scaler_jump3 = load('model/scaler_jump3.joblib')

    clf_zhongdian2 = load('model/trained_model_zhongdian2.joblib')
    scaler_zhongdian2 = load('model/scaler_zhongdian2.joblib')

    # 构建图标路径
    icon_path_jump0 = os.path.join('icon', 'jump0-1.png')
    icon_path_jump1 = os.path.join('icon', 'jump1-1.png')
    icon_path_jump2 = os.path.join('icon', 'jump2-1.png')
    icon_path_jump3 = os.path.join('icon', 'jump3-1.png')
    icon_path_zhongdian2 = os.path.join('icon', 'zhongdian2-1.png')

    template_jump0 = cv2.imread(icon_path_jump0, cv2.IMREAD_COLOR)
    template_jump1 = cv2.imread(icon_path_jump1, cv2.IMREAD_COLOR)
    template_jump2 = cv2.imread(icon_path_jump2, cv2.IMREAD_COLOR)
    template_jump3 = cv2.imread(icon_path_jump3, cv2.IMREAD_COLOR)
    template_zhongdian2 = cv2.imread(icon_path_zhongdian2, cv2.IMREAD_COLOR)

    template_gray_jump0 = cv2.cvtColor(template_jump0, cv2.COLOR_BGR2GRAY)
    template_gray_jump1 = cv2.cvtColor(template_jump1, cv2.COLOR_BGR2GRAY)
    template_gray_jump2 = cv2.cvtColor(template_jump2, cv2.COLOR_BGR2GRAY)
    template_gray_jump3 = cv2.cvtColor(template_jump3, cv2.COLOR_BGR2GRAY)
    template_gray_zhongdian2 = cv2.cvtColor(template_zhongdian2, cv2.COLOR_BGR2GRAY)

    w_jump0, h_jump0 = template_gray_jump0.shape[::-1]
    w_jump1, h_jump1 = template_gray_jump1.shape[::-1]
    w_jump2, h_jump2 = template_gray_jump2.shape[::-1]
    w_jump3, h_jump3 = template_gray_jump3.shape[::-1]
    w_zhongdian2, h_zhongdian2 = template_gray_zhongdian2.shape[::-1]

    # 设置需要捕获的屏幕区域
    fx,fy=pyautogui.size()
    x0, y0, width0, height0 = 1300, 50, 900, 1000
    region0=(x0,y0,width0,height0)
    x1, y1, width1, height1 = 1380, 50, 320, 650
    region1=(x1,y1,width1,height1)
    x2, y2, width2, height2 = 50, 150, 400, 500
    region2=(x2,y2,width2,height2)
    
    #time.sleep(2)  # 等待窗口打开

    #开始跳跃出发
    attempts = 0
    max_attempts=10
    icon_found_and_clicked = False
    while attempts < max_attempts and not icon_found_and_clicked:
        pane0_image = capture_screen_area(region0)
        pane0_gray = cv2.cvtColor(pane0_image, cv2.COLOR_BGR2GRAY)
        res_jump0 = cv2.matchTemplate(pane0_gray, template_gray_jump0, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res_jump0 >= threshold)

        for pt in zip(*loc[::-1]):  # Switch x and y positions
            center_x_jump0 = x0 + pt[0] + w_jump0 // 2  # 加上区域的起始X坐标
            center_y_jump0 = y0 + pt[1] + h_jump0 // 2  # 加上区域的起始Y坐标
            icon_image_jump0 = pane0_image[pt[1]:pt[1]+h_jump0, pt[0]:pt[0]+w_jump0]

            if predict_icon_status(icon_image_jump0, clf_jump0, scaler_jump0):
                print(f"找到下一站(黄门)...")
                pyautogui.moveTo(center_x_jump0, center_y_jump0)
                pyautogui.leftClick()
                pyautogui.leftClick()
                icon_found_and_clicked = True
                break  # Break the inner loop as we only need one active icon

        if not icon_found_and_clicked:
            print("未找到下一站(黄标门), Retrying...")
            attempts += 1
            scollscreen()
            #pyautogui.scroll(100)
            time.sleep(1)

    if not icon_found_and_clicked:
        print("未找到下一站(黄标门), after maximum attempts.")
 
    while True:

        panel_image = capture_screen_area(region1)
        panel_gray = cv2.cvtColor(panel_image, cv2.COLOR_BGR2GRAY)

        # 停靠空间站
        res_jump3 = cv2.matchTemplate(panel_gray, template_gray_jump3, cv2.TM_CCOEFF_NORMED)
        _, max_val_jump3, _, max_loc_jump3 = cv2.minMaxLoc(res_jump3)

        if max_val_jump3 > 0.8:
            top_left_jump3 = max_loc_jump3
            center_x_jump3 = x1 + top_left_jump3[0] + w_jump3 // 2
            center_y_jump3 = y1 + top_left_jump3[1] + h_jump3 // 2
            icon_image_jump3 = panel_image[top_left_jump3[1]:top_left_jump3[1]+h_jump3, top_left_jump3[0]:top_left_jump3[0]+w_jump3]

            if predict_icon_status(icon_image_jump3, clf_jump3, scaler_jump3):
                pyautogui.moveTo(center_x_jump3, center_y_jump3)
                pyautogui.click()
                print("准备停靠空间站...")
                print("Jump结束!")
                break  # 程序退出

        # 跳跃
        res_jump1 = cv2.matchTemplate(panel_gray, template_gray_jump1, cv2.TM_CCOEFF_NORMED)
        _, max_val_jump1, _, max_loc_jump1 = cv2.minMaxLoc(res_jump1)

        if max_val_jump1 > 0.8:  # 图标1识别成功
            top_left_jump1 = max_loc_jump1
            center_x_jump1 = x1 + top_left_jump1[0] + w_jump1 // 2
            center_y_jump1 = y1 + top_left_jump1[1] + h_jump1 // 2
            icon_image_jump1 = panel_image[top_left_jump1[1]:top_left_jump1[1]+h_jump1, top_left_jump1[0]:top_left_jump1[0]+w_jump1]

            if predict_icon_status(icon_image_jump1, clf_jump1, scaler_jump1):
                print(f"准备跳跃...")
                pyautogui.moveTo(center_x_jump1, center_y_jump1)
                pyautogui.click()
                time.sleep(3)
            else:
                # 跃迁
                res_jump2 = cv2.matchTemplate(panel_gray, template_gray_jump2, cv2.TM_CCOEFF_NORMED)
                _, max_val_jump2, _, max_loc_jump2 = cv2.minMaxLoc(res_jump2)
                if max_val_jump2 > 0.8:
                    top_left_jump2 = max_loc_jump2
                    center_x_jump2 = x1 + top_left_jump2[0] + w_jump2 // 2
                    center_y_jump2 = y1 + top_left_jump2[1] + h_jump2 // 2
                    print(f"准备跃迁...")
                    pyautogui.moveTo(center_x_jump2, center_y_jump2)
                    pyautogui.click()

        time.sleep(3)  # 每三秒执行一次检查

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()