#!/usr/bin/env python3
"""
s01_agent_loop.py - 智能体循环 (The Agent Loop)

AI 编码智能体的核心秘密就在这一个模式中:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (循环继续)

这是核心循环: 将工具结果反馈给模型, 直到模型决定停止。
生产级智能体会在此基础上叠加策略、钩子和生命周期控制。
"""

import os
import subprocess

from anthropic import Anthropic
from dotenv import load_dotenv

# 加载环境变量, override=True 确保覆盖已存在的变量
load_dotenv(override=True)

# 如果配置了自定义 API 地址, 移除默认的认证 token
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# 初始化 Anthropic 客户端
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# 系统提示词: 定义智能体的角色和工作目录
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# 工具定义: 仅提供一个 bash 工具
# 这是智能体的唯一能力 - 执行 shell 命令
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


def run_bash(command: str) -> str:
    """
    执行 bash 命令并返回结果。

    这是一个安全的命令执行函数, 包含以下安全措施:
    1. 危险命令拦截: 阻止可能破坏系统的命令
    2. 超时保护: 命令执行超过 120 秒自动终止
    3. 输出截断: 限制输出长度防止内存溢出

    Args:
        command: 要执行的 shell 命令字符串

    Returns:
        str: 命令执行结果 (stdout + stderr), 或错误信息
    """
    # 危险命令黑名单: 阻止可能导致系统损坏的命令
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        # 执行命令, 捕获 stdout 和 stderr
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        # 限制输出长度为 50000 字符, 防止上下文过长
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def agent_loop(messages: list):
    """
    智能体核心循环: 持续调用 LLM 直到模型决定停止。

    这是整个智能体的心脏, 工作流程如下:
    1. 调用 LLM API 获取响应
    2. 将助手响应追加到消息历史
    3. 检查 stop_reason:
       - 如果不是 "tool_use", 说明模型完成任务, 退出循环
       - 如果是 "tool_use", 执行工具调用并将结果追加到消息历史
    4. 回到步骤 1 继续循环

    这个循环会持续运行, 直到:
    - 模型决定不再调用工具 (任务完成)
    - 模型给出 end_turn 或其他停止原因

    Args:
        messages: 消息历史列表, 包含 user 和 assistant 的对话记录
    """
    while True:
        # 调用 LLM API, 传入消息历史和工具定义
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        # 追加助手响应到消息历史 (保持对话完整性)
        messages.append({"role": "assistant", "content": response.content})

        # 检查停止原因: 如果模型没有调用工具, 说明任务完成
        if response.stop_reason != "tool_use":
            return

        # 执行所有工具调用, 收集结果
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # 打印正在执行的命令 (黄色高亮)
                print(f"\033[33m$ {block.input['command']}\033[0m")
                output = run_bash(block.input["command"])
                # 打印命令输出的前 200 字符 (便于观察)
                print(output[:200])
                # 构造工具结果消息, 必须包含 tool_use_id 以匹配原始调用
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output})

        # 将工具结果作为 user 消息追加, 供下一轮循环使用
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    """
    主程序入口: 提供交互式命令行界面。

    运行方式:
        python agents/s01_agent_loop.py

    支持的命令:
        - 输入任意问题让智能体执行
        - 输入 'q' 或 'exit' 或空行退出程序
    """
    history = []  # 消息历史, 跨多轮对话保持上下文

    while True:
        try:
            # 读取用户输入 (青色提示符)
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        # 退出条件: q, exit, 或空行
        if query.strip().lower() in ("q", "exit", ""):
            break

        # 将用户问题追加到历史
        history.append({"role": "user", "content": query})

        # 运行智能体循环
        agent_loop(history)

        # 打印最终响应 (如果是文本块)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
