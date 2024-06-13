import sys
import time
import logging
import subprocess

# 配置日志
logging.basicConfig(filename='go.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def run_script(script_name):
    try:
        logging.info(f'开始执行: {script_name}')
        result = subprocess.run(['python', script_name], capture_output=True, text=True)
        logging.info(f'{script_name} 输出: {result.stdout}')
        if result.returncode != 0:
            logging.error(f'{script_name} 执行失败: {result.stderr}')
            return False
        return True
    except Exception as e:
        logging.error(f'运行 {script_name} 时出错: {e}')
        return False

def main():
    scripts = ['task.py', 'jump1.py', 'talk.py', 'navigate.py', 'jump2.py']  # 定义脚本列表

    # 检查命令行参数是否指定了开始脚本
    starting_script = 0  # 默认从第一个脚本开始
    if len(sys.argv) > 1:
        try:
            starting_script = int(sys.argv[1]) - 1  # 用户可能从1开始计数，而脚本数组是从0开始
            if starting_script < 0 or starting_script >= len(scripts):
                print(f"Invalid script index {starting_script + 1}. Please enter a valid index between 1 and {len(scripts)}.")
                return
        except ValueError:
            print("Invalid input. Please enter a numerical index.")
            return

    # 无限循环执行脚本
    while True:
        for i in range(starting_script, len(scripts)):
            if not run_script(scripts[i]):
                logging.error(f'Failed to execute {scripts[i]}')
                break
        logging.info('完成一个完整任务流程')
        # 可选择休眠一段时间再重复执行整个流程
        time.sleep(5)  # 假设每次循环之间等待5秒

if __name__ == '__main__':
    main()
