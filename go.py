import sys
import time
import logging
import subprocess

# 配置日志
logging.basicConfig(filename='task.log', level=logging.DEBUG,  # 更改日志级别为DEBUG以记录更多信息
                    format='%(asctime)s - %(levelname)s - %(message)s')

def run_script(script_name):
    try:
        logging.info(f'开始执行: {script_name}')
        result = subprocess.run(['python', script_name], capture_output=True, text=True)
        if result.stdout:
            logging.info(f'{script_name} 标准输出: {result.stdout}')
        if result.stderr:
            logging.error(f'{script_name} 错误输出: {result.stderr}')

        if result.returncode != 0:
            logging.error(f'{script_name} 执行失败，返回码: {result.returncode}')
            return False
        return True
    except Exception as e:
        logging.error(f'运行 {script_name} 时出错: {str(e)}')
        return False

def main():
    scripts = ['task.py', 'jump1.py', 'talk.py', 'navigate.py', 'jump2.py']

    # 检查命令行参数是否指定了开始脚本
    starting_script = 0
    if len(sys.argv) > 1:
        try:
            starting_script = int(sys.argv[1]) - 1
            if starting_script < 0 or starting_script >= len(scripts):
                logging.error(f"无效的脚本索引 {starting_script + 1}")
                sys.exit(f"请输入一个在1到{len(scripts)}之间的索引")
        except ValueError:
            logging.error("输入的索引非数字")
            sys.exit("请输入一个数字索引")

    current_script = starting_script
    while True:
        if not run_script(scripts[current_script]):
            logging.error(f'{scripts[current_script]} 执行失败，停止运行')
            break
        logging.info(f'已完成 {scripts[current_script]}')
        
        # Move to the next script, or wrap around to the first script
        current_script += 1
        if current_script >= len(scripts):
            current_script = 0  # Reset to the first script if we reach the end of the list

        # 如果回到了起始点，添加一个等待时间再开始新一轮
        if current_script == 0:
            logging.info('完成一个完整任务流程，准备重启')
            time.sleep(5)  # 每次循环之间的等待时间



if __name__ == '__main__':
    main()
