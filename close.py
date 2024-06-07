# close.py
import cv2
import numpy as np
import pyautogui
import time
from joblib import load
import os

def capture_full_screen():
    screenshot = pyautogui.screenshot()
    screen_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screen_image

def find_and_close_icons(template_path, clf, scaler, w, h):
    screen = capture_full_screen()
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray_screen, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)

    if loc[0].size > 0:
        for pt in zip(*loc[::-1]):  # Switch columns and rows
            global_x = pt[0] + w // 2
            global_y = pt[1] + h // 2
            pyautogui.moveTo(global_x+11, global_y)
            pyautogui.click()
            print(f"Clicked close icon at: ({global_x}, {global_y})")
        return True
    return False

def close_icons_main():
    # Load models and scalers
    clf_close1 = load('trained_model_close1.joblib')
    scaler_close1 = load('scaler_close1.joblib')

    # Define paths to icon templates
    icon_path_close1 = os.path.join('icon', 'close1-1.png')

    # Load and process icon templates
    template_close1 = cv2.imread(icon_path_close1, cv2.IMREAD_COLOR)
    template_gray_close1 = cv2.cvtColor(template_close1, cv2.COLOR_BGR2GRAY)
    w_close1, h_close1 = template_gray_close1.shape[::-1]

    find_and_close_icons(icon_path_close1, clf_close1, scaler_close1, w_close1, h_close1)

if __name__ == "__main__":
    close_icons_main()
