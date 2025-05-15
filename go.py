import subprocess
import time
import logging
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('task.log', mode='a'),
        logging.StreamHandler()
    ]
)

def run_script(script_name):
    """运行指定脚本并捕获输出"""
    logging.info(f"开始执行: {script_name}")
    try:
        cmd = ['python'] + script_name.split()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logging.error(f"{script_name} 执行失败，返回码: {result.returncode}")
            return False
        return True
    except Exception as e:
        logging.error(f"{script_name} 执行失败: {e}")
        return False

def main():
    # 定义完整脚本列表
    scripts = ['task.py', 'jump1.py', 'talk.py', 'navigate.py', 'jump1.py']
    
    # 获取用户指定的开始位置（1-based 索引，转换为 0-based）
    if len(sys.argv) > 1:
        try:
            start_index = int(sys.argv[1]) - 1
            if start_index < 0 or start_index >= len(scripts):
                logging.error(f"开始位置 {sys.argv[1]} 超出范围（1-{len(scripts)}）")
                sys.exit(1)
        except ValueError:
            logging.error(f"无效的开始位置: {sys.argv[1]}")
            sys.exit(1)
    else:
        start_index = 0

    cycle_count = 0
    first_cycle = True  # 标记第一次循环

    while True:
        cycle_count += 1
        if first_cycle:
            # 第一次循环从用户指定位置开始
            current_scripts = scripts[start_index:]
            logging.info(f"开始第 {cycle_count} 次循环，脚本列表: {current_scripts}")
            first_cycle = False
        else:
            # 后续循环使用完整列表
            current_scripts = scripts
            logging.info(f"开始第 {cycle_count} 次循环，脚本列表: {current_scripts}")

        for script in current_scripts:
            if not run_script(script):
                logging.error(f"脚本 {script} 失败，终止当前循环")
                return
            time.sleep(1)

        logging.info(f"第 {cycle_count} 次循环完成，所有脚本执行成功")
        logging.info("等待 1 秒后开始下一轮循环")
        time.sleep(10)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("程序被用户中断")
        sys.exit(0)