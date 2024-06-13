import cv2
import numpy as np
import pyautogui
import pytesseract
import time
from joblib import load
# 在 talk.py 中添加对 close.py 的调用
from close import close_icons_main

import os

def capture_full_screen():
    """Capture the entire screen and convert it to OpenCV format."""
    screenshot = pyautogui.screenshot()
    screen_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screen_image

def predict_icon_status(image, clf, scaler):
    """Use a machine learning model to determine the status of an icon."""
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist_scaled = scaler.transform([hist.flatten()])
    prediction = clf.predict(hist_scaled)[0]
    return prediction == 1  # Returns True if the icon is active

def main():
    # Load models and scalers
    clf_talk1 = load('model/trained_model_talk1.joblib')
    scaler_talk1 = load('model/scaler_talk1.joblib')
    clf_talk2 = load('model/trained_model_talk2.joblib')
    scaler_talk2 = load('model/scaler_talk2.joblib')
    clf_out1 = load('model/trained_model_out1.joblib')
    scaler_out1 = load('model/scaler_out1.joblib')

    # Define paths to icon templates
    icon_path_talk1 = os.path.join('icon', 'talk1-1.png')
    icon_path_talk2 = os.path.join('icon', 'talk2-1.png')
    icon_path_out1 = os.path.join('icon', 'out1-1.png')

    # Load and process icon templates
    template_talk1 = cv2.imread(icon_path_talk1, cv2.IMREAD_COLOR)
    template_gray_talk1 = cv2.cvtColor(template_talk1, cv2.COLOR_BGR2GRAY)
    w_talk1, h_talk1 = template_gray_talk1.shape[::-1]
    template_talk2 = cv2.imread(icon_path_talk2, cv2.IMREAD_COLOR)
    template_gray_talk2 = cv2.cvtColor(template_talk2, cv2.COLOR_BGR2GRAY)
    w_talk2, h_talk2 = template_gray_talk2.shape[::-1]
    template_out1 = cv2.imread(icon_path_out1, cv2.IMREAD_COLOR)
    template_gray_out1 = cv2.cvtColor(template_out1, cv2.COLOR_BGR2GRAY)
    w_out1, h_out1 = template_gray_out1.shape[::-1]

    check_talk1 = True

    while True:
        if check_talk1:
            screen = capture_full_screen()
            gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            res_talk1 = cv2.matchTemplate(gray_screen, template_gray_talk1, cv2.TM_CCOEFF_NORMED)
            max_val_talk1, max_loc_talk1 = cv2.minMaxLoc(res_talk1)[1], cv2.minMaxLoc(res_talk1)[3]

            if max_val_talk1 > 0.8:
                if predict_icon_status(screen[max_loc_talk1[1]:max_loc_talk1[1]+h_talk1, max_loc_talk1[0]:max_loc_talk1[0]+w_talk1], clf_talk1, scaler_talk1):
                    pyautogui.moveTo(max_loc_talk1[0] + w_talk1 // 2, max_loc_talk1[1] + h_talk1 // 2)
                    pyautogui.click()
                    check_talk1 = False  # Stop checking for talk1-1 icon
                    time.sleep(3)  # Wait for the dialog to appear
                    screen = capture_full_screen()
                    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
                    if "目标完成" in pytesseract.image_to_string(screen, lang='chi_sim+eng'):
                        print("目标完成 detected. Checking for talk2-1 icon.")
                        res_talk2 = cv2.matchTemplate(gray_screen, template_gray_talk2, cv2.TM_CCOEFF_NORMED)
                        max_val_talk2, max_loc_talk2 = cv2.minMaxLoc(res_talk2)[1], cv2.minMaxLoc(res_talk2)[3]
                        if max_val_talk2 > 0.8:
                            pyautogui.moveTo(max_loc_talk2[0] + w_talk2 // 2, max_loc_talk2[1] + h_talk2 // 2)
                            pyautogui.click()
                            print("Talk2 icon clicked successfully. Checking for out1 icon.")
                            time.sleep(1)

                            print("Performed right click, now closing icons.")
                            # 执行关闭窗口操作
                            close_icons_main()
                            # 再按下 Ctrl+W
                            pyautogui.hotkey('ctrl', 'w')
                            break
                    else:
                        print("任务 not completed. Retrying...")
                else:
                    print("对话窗口 is inactive.")
            else:
                print("开始对话 not found.")
        time.sleep(1)  # Check every second

    print("Program ended.")

if __name__ == "__main__":
    main()
