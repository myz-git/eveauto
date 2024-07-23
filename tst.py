import pyautogui
import pyperclip  # 导入 pyperclip
import time
import keyboard





def main():
    print(f"准备搬运")
    time.sleep(3)
    fx, fy = pyautogui.size()
    region = (0, 0, fx, fy)
    x = region[2] //2
    y = region[3] //2
    pyautogui.moveTo(x, y)
    pyautogui.leftClick
    #pyautogui.hotkey('alt', 'c')
    keyboard.send('Alt+C')

if __name__ == "__main__":
    main()