#!/usr/bin/env python3
"""
s04_subagent.py - 子智能体 (Subagents)

启动一个子智能体, 使用全新的 messages=[] 空列表。
子智能体在自己的上下文中工作, 共享文件系统,
但只返回摘要给父智能体。

    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- 全新上下文
    |                  |  dispatch   |                  |
    | tool: task       | ---------->| while tool_use:  |
    |   prompt="..."   |            |   call tools     |
    |   description="" |            |   append results |
    |                  |  summary   |                  |
    |   result = "..." | <--------- | return last text |
    +------------------+             +------------------+
              |
    父智能体上下文保持干净
    子智能体上下文被丢弃

关键洞察: "进程隔离天然带来上下文隔离。"
"""

import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# 父智能体系统提示词: 强调使用 task 工具委托子任务
SYSTEM = f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."

# 子智能体系统提示词: 强调完成后要总结结果
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."


# ============== 工具实现函数 (父/子智能体共享) ==============

def safe_path(p: str) -> Path:
    """
    安全路径解析: 确保路径不会逃逸出工作目录。

    Args:
        p: 用户提供的文件路径

    Returns:
        Path: 安全的绝对路径对象

    Raises:
        ValueError: 如果路径试图逃逸工作目录
    """
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """
    执行 bash 命令并返回结果。

    Args:
        command: 要执行的 shell 命令

    Returns:
        str: 命令输出或错误信息
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    """
    读取文件内容。

    Args:
        path: 文件路径
        limit: 可选的行数限制

    Returns:
        str: 文件内容或错误信息
    """
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """
    写入文件内容。

    Args:
        path: 文件路径
        content: 要写入的内容

    Returns:
        str: 操作结果描述
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    编辑文件: 替换首次匹配的文本。

    Args:
        path: 文件路径
        old_text: 要替换的原始文本
        new_text: 替换后的新文本

    Returns:
        str: 操作结果描述
    """
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# 工具分发映射表
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# 子智能体工具列表: 包含所有基础工具, 但没有 task (禁止递归生成子智能体)
# 这是上下文隔离的关键: 子智能体不能创建子智能体
CHILD_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]


def run_subagent(prompt: str) -> str:
    """
    运行子智能体: 使用全新的上下文执行任务, 只返回摘要。

    这是上下文隔离的核心机制:
    1. 子智能体以空的 messages=[] 启动 (全新上下文)
    2. 子智能体可以自由调用工具, 累积自己的消息历史
    3. 子智能体最多运行 30 轮 (安全限制)
    4. 任务完成后, 子智能体的整个消息历史被丢弃
    5. 只有最终的文本摘要返回给父智能体

    这意味着子智能体可以读取 100 个文件,
    但父智能体只收到一句话的摘要,
    大大节省了父智能体的上下文空间。

    Args:
        prompt: 子智能体要执行的任务描述

    Returns:
        str: 子智能体的最终摘要文本
    """
    # 全新的消息历史 - 这是上下文隔离的关键
    sub_messages = [{"role": "user", "content": prompt}]

    # 安全限制: 最多 30 轮工具调用
    for _ in range(30):
        response = client.messages.create(
            model=MODEL, system=SUBAGENT_SYSTEM, messages=sub_messages,
            tools=CHILD_TOOLS, max_tokens=8000,
        )
        sub_messages.append({"role": "assistant", "content": response.content})

        # 如果没有工具调用, 任务完成
        if response.stop_reason != "tool_use":
            break

        # 执行工具调用
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                # 限制输出长度, 防止子智能体上下文溢出
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)[:50000]})

        sub_messages.append({"role": "user", "content": results})

    # 只返回最终的文本摘要 - 子智能体的整个上下文被丢弃
    return "".join(b.text for b in response.content if hasattr(b, "text")) or "(no summary)"


# 父智能体工具列表: 基础工具 + task 工具
# task 工具是父智能体独有的, 用于委托子任务
PARENT_TOOLS = CHILD_TOOLS + [
    {"name": "task", "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
     "input_schema": {"type": "object", "properties": {"prompt": {"type": "string"}, "description": {"type": "string", "description": "Short description of the task"}}, "required": ["prompt"]}},
]


def agent_loop(messages: list):
    """
    父智能体核心循环: 处理 task 工具调用时启动子智能体。

    新增功能:
    1. 支持 task 工具 - 委托子任务给子智能体
    2. 子智能体运行在独立上下文中
    3. 只将摘要结果追加到父智能体的消息历史

    Args:
        messages: 父智能体的消息历史列表
    """
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=PARENT_TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                # 特殊处理 task 工具: 启动子智能体
                if block.name == "task":
                    desc = block.input.get("description", "subtask")
                    print(f"> task ({desc}): {block.input['prompt'][:80]}")
                    output = run_subagent(block.input["prompt"])
                else:
                    # 其他工具直接调用处理函数
                    handler = TOOL_HANDLERS.get(block.name)
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"

                print(f"  {str(output)[:200]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    """
    主程序入口: 提供交互式命令行界面。

    运行方式:
        python agents/s04_subagent.py

    新增特性:
        - 父智能体可以使用 task 工具委托子任务
        - 子智能体在独立上下文中运行, 不污染父智能体的对话历史
    """
    history = []
    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
