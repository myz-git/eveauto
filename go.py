import subprocess
import time

def run_script(script_name):
    """运行给定的Python脚本"""
    try:
        # 执行Python脚本并等待其完成
        completed = subprocess.run(['python', script_name], check=True)
        print(f"{script_name} executed successfully: {completed}")
    except subprocess.CalledProcessError as e:
        # 如果脚本执行失败，打印错误
        print(f"Error executing {script_name}: {e}")

def main():
    # 按顺序执行 jump.py, talk.py, jump.py
    print("Starting the script sequence...")
    run_script('jump.py')
    run_script('talk.py')
    run_script('navigate.py')
    
    time.sleep(5)
    run_script('jump.py')
    time.sleep(1)
    run_script('task.py')
    print("Script sequence completed.")

if __name__ == "__main__":
    main()
