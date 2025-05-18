import subprocess
import time
import logging
import sys
import argparse
import os
from dbcfg import list_agents, getaddr, get_location_name

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('task.log', mode='a'),
        logging.StreamHandler()
    ]
)

def run_script(script_name, agent_id):
    """运行指定脚本并捕获输出，传递 agent_id"""
    logging.info(f"开始执行: {script_name} (agent_id={agent_id})")
    try:
        os.environ['AGENT_ID'] = str(agent_id)
        cmd = ['python'] + script_name.split()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logging.error(f"{script_name} 执行失败，返回码: {result.returncode}")
            return False
        return True
    except Exception as e:
        logging.error(f"{script_name} 执行失败: {e}")
        return False
    finally:
        os.environ.pop('AGENT_ID', None)

def select_agent():
    """交互式选择代理人"""
    agents = list_agents()
    if not agents:
        logging.error("数据库中没有代理人记录")
        sys.exit(1)
    
    print("\n可用代理人列表：")
    for agent in agents:
        agent_id, agent_name, addr, info = agent
        print(f"ID: {agent_id}, 代理人: {agent_name}, 空间站: {addr}, 信息: {info or '无'}")
    
    while True:
        try:
            choice = input("\n请输入代理人 ID: ")
            agent_id = int(choice)
            if any(agent[0] == agent_id for agent in agents):
                agent_info = getaddr(agent_id)
                if agent_info:
                    logging.info(f"选择代理人 ID={agent_id}, 代理人={agent_info[0]}, 空间站={agent_info[1]}")
                    return agent_id
                else:
                    print(f"无效的代理人 ID: {agent_id}")
            else:
                print(f"无效的代理人 ID: {agent_id}")
        except ValueError:
            print("请输入有效的数字 ID")

def main():
    # 定义完整脚本列表
    scripts = ['task.py', 'jump1.py', 'talk.py', 'navigate.py', 'jump1.py']
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="EVE Auto Task Runner")
    parser.add_argument('--agent', type=int, help='Agent ID to use')
    parser.add_argument('--start', type=int, default=1, help='Starting script position (1-based index, 1 to %d)' % len(scripts))
    args = parser.parse_args()

    # 选择 agent_id
    if args.agent is not None:
        agent_id = args.agent
        if not getaddr(agent_id):
            logging.error(f"无效的 agent_id: {agent_id}")
            sys.exit(1)
    else:
        agent_id = select_agent()

    # 验证启动位置
    start_index = args.start - 1  # 转换为 0-based 索引
    if start_index < 0 or start_index >= len(scripts):
        logging.error(f"开始位置 {args.start} 超出范围（1-{len(scripts)}）")
        sys.exit(1)

    cycle_count = 0
    first_cycle = True

    while True:
        cycle_count += 1
        if first_cycle:
            current_scripts = scripts[start_index:]
            logging.info(f"开始第 {cycle_count} 次循环，脚本列表: {current_scripts}, agent_id={agent_id}, start_index={start_index + 1}")
            first_cycle = False
        else:
            current_scripts = scripts
            logging.info(f"开始第 {cycle_count} 次循环，脚本列表: {current_scripts}, agent_id={agent_id}, start_index=1")

        for script in current_scripts:
            if not run_script(script, agent_id):
                logging.error(f"脚本 {script} 失败，终止当前循环")
                return
            time.sleep(1)

        logging.info(f"第 {cycle_count} 次循环完成，所有脚本执行成功")
        logging.info("等待 10 秒后开始下一轮循环")
        time.sleep(10)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("程序被用户中断")
        sys.exit(0)