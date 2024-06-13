# outsite.py
import cv2
import numpy as np
import pyautogui
import time
from joblib import load
import os

def scollscreen(max_attempts=10):
    """转动屏幕"""
    fx,fy=pyautogui.size()
    pyautogui.moveTo(fx/2,fy/2+270,0.2)
    pyautogui.dragRel(-50,0,0.4,pyautogui.easeOutQuad)    


def capture_full_screen():
    screenshot = pyautogui.screenshot()
    screen_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screen_image

def predict_icon_status(image, clf, scaler):
    """Use a machine learning model to determine the status of an icon."""
    # 将图像转换为适合模型输入的格式
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist_scaled = scaler.transform([hist.flatten()])
    prediction = clf.predict(hist_scaled)[0]
    return prediction == 1  # Returns True if the icon is active

def find_and_click_icon(template, width, height, clf, scaler, max_attempts=3,offset_x=0,offset_y=0):
    attempts = 0
    while attempts < max_attempts:
        screen = capture_full_screen()
        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_screen, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > 0.8:
            icon_image = screen[max_loc[1]:max_loc[1]+height, max_loc[0]:max_loc[0]+width]
            if predict_icon_status(icon_image, clf, scaler):
                x=max_loc[0] + width // 2+offset_x
                y=max_loc[1] + height // 2+offset_y
                pyautogui.moveTo(x, y)
                pyautogui.leftClick()
                print("检测到离站 ,开始出站...")
 
                return True
        attempts += 1
        scollscreen()
        print(f"Attempt {attempts}/{max_attempts}: Icon not found, retrying...")
        time.sleep(1)
    
    print("Failed to find icon after maximum attempts. Exiting...")
    exit(1)

def outsite_icons_main():
    # Load models and scalers
    clf_out1 = load('model/trained_model_out1.joblib')
    scaler_out1 = load('model/scaler_out1.joblib')

    # Define paths to icon templates
    icon_path_out1 = os.path.join('icon', 'out1-1.png')

    # Load and process icon templates
    template_out1 = cv2.imread(icon_path_out1, cv2.IMREAD_COLOR)
    template_gray_out1 = cv2.cvtColor(template_out1, cv2.COLOR_BGR2GRAY)
    w_out1, h_out1 = template_gray_out1.shape[::-1]

    find_and_click_icon(template_gray_out1, w_out1, h_out1, clf_out1, scaler_out1)  
               
if __name__ == "__main__":
    outsite_icons_main()
