import pyautogui
import pyperclip  # 导入 pyperclip
import time
import keyboard
import pynput


"""
https://www.cnblogs.com/tobe-goodlearner/p/tutorial-pynput.html
"""

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
    #keyboard.send('Alt+C')
    
    
    """
    ctr = pynput.keyboard.Controller()
    ctr.press(pynput.keyboard.Key.alt)
    ctr.press('c')
    time.sleep(0.1)
    ctr.release(pynput.keyboard.Key.alt)
    ctr.release('c')
    """
    #使用with封装 ,效果是进入语句块时顺序按下提供按键，退出语句块时逆序释放按键。
    #打开仓库(alt+c)
    ctr = pynput.keyboard.Controller()
    with ctr.pressed(
            pynput.keyboard.Key.alt,
            'c'):
        time.sleep(0.5)  #这里一定要加个sleep 否则看不到按键结果
        pass
    
    time.sleep(1)
    #关闭窗口(ctrl+w)
    with ctr.pressed(
            pynput.keyboard.Key.ctrl,
            'w'):
        time.sleep(0.5)
        pass    


if __name__ == "__main__":
    main()