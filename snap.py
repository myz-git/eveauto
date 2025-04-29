import cv2
import numpy as np
import pyautogui
import time
import os
import sys
from glob import glob

def capture_screen():
    """捕获整个屏幕的截图并返回，转换屏幕截图为OpenCV可处理的BGR格式。"""
    screenshot = pyautogui.screenshot()
    screen_image = np.array(screenshot)
    screen_image = cv2.cvtColor(screen_image, cv2.COLOR_RGB2BGR)
    return screen_image

def find_and_save_icon(template_filename, save_folder, capture_interval=1, num_captures=100):
    """使用模板匹配技术在屏幕截图中查找图标，显示候选区域并手动标注后保存。"""
    template_path = os.path.join('icon', template_filename)  # 构建模板路径
    base_filename = os.path.splitext(template_filename)[0]  # 从文件名中提取基本名字，不含扩展名

    # 创建 jump1-1 和 jump1-0 的保存文件夹
    jump1_1_folder = os.path.join(save_folder, 'jump1-1')
    jump1_0_folder = os.path.join(save_folder, 'jump1-0')
    if not os.path.exists(jump1_1_folder):
        os.makedirs(jump1_1_folder)
    if not os.path.exists(jump1_0_folder):
        os.makedirs(jump1_0_folder)

    # 计算 jump1-1 和 jump1-0 文件夹中已存在的文件序号
    existing_jump1_1 = glob(os.path.join(jump1_1_folder, f'jump1-1-*.png'))
    existing_jump1_0 = glob(os.path.join(jump1_0_folder, f'jump1-0-*.png'))
    max_index_1 = 0
    max_index_0 = 0
    if existing_jump1_1:
        max_index_1 = max([int(file.split('-')[-1].split('.')[0]) for file in existing_jump1_1])
    if existing_jump1_0:
        max_index_0 = max([int(file.split('-')[-1].split('.')[0]) for file in existing_jump1_0])

    save_count_1 = max_index_1 + 1  # jump1-1 文件编号
    save_count_0 = max_index_0 + 1  # jump1-0 文件编号

    # 加载模板图标
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"模板图像 '{template_path}' 无法加载")
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]

    for _ in range(num_captures):
        screen_image = capture_screen()
        screen_gray = cv2.cvtColor(screen_image, cv2.COLOR_BGR2GRAY)
        
        # 模板匹配
        res = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val > 0.8:  # 如果匹配度高于0.8，显示并手动标注
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            
            # 截取图标区域
            icon_image = screen_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            
            # 显示候选区域供用户标注
            cv2.imshow("Candidate", icon_image)
            print(f"坐标: ({top_left[0]}, {top_left[1]})")
            print("按 1 表示 jump1-1，按 0 表示 jump1-0，按 ESC 跳过")
            key = cv2.waitKey(0) & 0xFF

            # 根据用户输入保存到对应文件夹
            if key == ord('1'):
                save_path = os.path.join(jump1_1_folder, f'jump1-1-{save_count_1}.png')
                cv2.imwrite(save_path, icon_image)
                print(f"Saved: {save_path}")
                save_count_1 += 1
            elif key == ord('0'):
                save_path = os.path.join(jump1_0_folder, f'jump1-0-{save_count_0}.png')
                cv2.imwrite(save_path, icon_image)
                print(f"Saved: {save_path}")
                save_count_0 += 1
            else:
                print("Skipped")
            
        else:
            print("Icon not found")

        cv2.destroyAllWindows()
        time.sleep(capture_interval)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python snap.py jump1-1.png")
        sys.exit(1)

    template_filename = sys.argv[1]  # 从命令行参数获取图标文件名
    save_folder = 'traindata'  # 指定保存图标图像的根文件夹
    find_and_save_icon(template_filename, save_folder)