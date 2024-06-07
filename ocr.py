import pyautogui
import pytesseract
from PIL import Image
import time

def find_and_move_to_text(text, region=None):
    while True:
        # 截取屏幕指定区域
        if region:
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()

        # 将截图转换为图像数据，以便更详细地处理识别结果
        data = pytesseract.image_to_boxes(screenshot, lang='chi_sim', config='--psm 10')
        print(data)
        # 检查是否找到特定的文字
        if text in data:
                # 计算文字的中心位置
                x = data['left'] + data['width'] // 2
                y = data['top']+ data['height']// 2
                # 如果使用了区域截图，确保加上区域的原点坐标
                if region:
                    x += region[0]
                    y += region[1]
                # 移动鼠标到文字位置
                pyautogui.moveTo(x, y)
                print(f"鼠标已移动到位置：({x}, {y})")
                pyautogui.rightClick()
                return
        print("未找到文字，继续扫描")
        time.sleep(1)

# 调用函数，寻找屏幕上的“搜索”文字
# 你可以通过设置 region 参数来指定搜索区域
find_and_move_to_text("基科德")
