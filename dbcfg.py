import sqlite3
import os
from utils import log_message

def init_db():
    """初始化数据库和 ageninfo 表"""
    try:
        conn = sqlite3.connect('tasks.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agentinfo (
                agent_id INTEGER PRIMARY KEY,
                agent TEXT NOT NULL,
                addr TEXT NOT NULL,
                info TEXT
            )
        ''')
        # 检查是否已有数据，若无则插入示例记录
        cursor.execute('SELECT COUNT(*) FROM agentinfo')
        if cursor.fetchone()[0] == 0:
            cursor.execute('''
                INSERT INTO ageninfo (agent_id, agent, addr, info)
                VALUES (?, ?, ?, ?)
            ''', (1, '克吉塔', '德勒斯 I - 卫星 1 - 自由扩张集团 储存工厂', '初始代理人'))
        conn.commit()
        log_message("INFO", "数据库 tasks.db 初始化成功")
    except sqlite3.Error as e:
        log_message("ERROR", f"数据库初始化失败: {e}")
        raise
    finally:
        conn.close()

def getaddr(agent_id):
    """根据 agent_id 查询 agent 和 addr"""
    try:
        conn = sqlite3.connect('tasks.db')
        cursor = conn.cursor()
        cursor.execute('SELECT agent, addr FROM agentinfo WHERE agent_id = ?', (agent_id,))
        result = cursor.fetchone()
        if result:
            log_message("INFO", f"查询 agent_id={agent_id} 成功: agent={result[0]}, addr={result[1]}")
            return result
        else:
            log_message("ERROR", f"未找到 agent_id={agent_id}")
            return None
    except sqlite3.Error as e:
        log_message("ERROR", f"数据库查询失败: {e}")
        return None
    finally:
        conn.close()

def get_location_name(tag):
    """从数据库加载代理人或空间站信息"""
    agent_id = os.environ.get('AGENT_ID')
    if not agent_id:
        log_message("ERROR", "未设置 AGENT_ID 环境变量")
        raise ValueError("未设置 AGENT_ID 环境变量")
    
    try:
        agent_id = int(agent_id)
        result = getaddr(agent_id)
        if not result:
            log_message("ERROR", f"未找到 agent_id={agent_id} 的记录")
            raise ValueError(f"未找到 agent_id={agent_id} 的记录")
        
        agent, addr = result
        if tag == 'agent':
            log_message("INFO", f"加载代理人名称: {agent}")
            return agent
        elif tag == 'addr':
            log_message("INFO", f"加载空间站地址: {addr}")
            return addr
        else:
            log_message("ERROR", f"无效的 tag: {tag}")
            raise ValueError(f"无效的 tag: {tag}")
    except ValueError as e:
        log_message("ERROR", f"加载位置信息失败: {e}")
        raise

def list_agents():
    """列出所有代理人信息"""
    try:
        conn = sqlite3.connect('tasks.db')
        cursor = conn.cursor()
        cursor.execute('SELECT agent_id, agent, addr, info FROM ageninfo')
        agents = cursor.fetchall()
        return agents
    except sqlite3.Error as e:
        log_message("ERROR", f"列出代理人失败: {e}")
        return []
    finally:
        conn.close()

# 初始化数据库
if not os.path.exists('tasks.db'):
    init_db()