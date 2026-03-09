#!/usr/bin/env python3
"""
s02_tool_use.py - 工具使用 (Tools)

s01 的智能体循环完全没变。我们只是在工具数组中添加了更多工具,
并用分发字典 (dispatch map) 来路由调用。

    +----------+      +-------+      +------------------+
    |   User   | ---> |  LLM  | ---> | Tool Dispatch    |
    |  prompt  |      |       |      | {                |
    +----------+      +---+---+      |   bash: run_bash |
                          ^          |   read: run_read |
                          |          |   write: run_wr  |
                          +----------+   edit: run_edit |
                          tool_result| }                |
                                     +------------------+

关键洞察: "循环完全不用改。只需要添加工具。"
"""

import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(override=True)

# 如果配置了自定义 API 地址, 移除默认的认证 token
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# 工作目录: 所有文件操作都限制在此目录下
WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# 系统提示词
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


def safe_path(p: str) -> Path:
    """
    安全路径解析: 确保路径不会逃逸出工作目录。

    这是文件操作的安全基石:
    1. 将相对路径解析为绝对路径
    2. 检查解析后的路径是否仍在工作目录内
    3. 如果路径逃逸, 抛出异常

    Args:
        p: 用户提供的文件路径 (可以是相对或绝对路径)

    Returns:
        Path: 安全的绝对路径对象

    Raises:
        ValueError: 如果路径试图逃逸工作目录 (如 ../../../etc/passwd)
    """
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """
    执行 bash 命令并返回结果。

    安全措施:
    1. 危险命令黑名单拦截
    2. 120 秒超时保护
    3. 输出长度限制

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
        path: 文件路径 (相对于工作目录)
        limit: 可选, 限制读取的行数

    Returns:
        str: 文件内容, 或错误信息
    """
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        # 如果指定了行数限制, 截断并添加提示
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """
    写入文件内容 (覆盖现有文件)。

    自动创建不存在的父目录。

    Args:
        path: 文件路径 (相对于工作目录)
        content: 要写入的内容

    Returns:
        str: 操作结果描述, 或错误信息
    """
    try:
        fp = safe_path(path)
        # 自动创建父目录
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    编辑文件: 替换文件中首次出现的文本。

    这是精确编辑操作, 只替换第一次匹配的文本,
    比 sed 更安全, 不会意外替换多处。

    Args:
        path: 文件路径 (相对于工作目录)
        old_text: 要替换的原始文本 (必须完全匹配)
        new_text: 替换后的新文本

    Returns:
        str: 操作结果描述, 或错误信息
    """
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        # 只替换第一次出现的位置 (count=1)
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# 工具分发映射表: {工具名: 处理函数}
# 这是工具调用的核心路由机制, 新增工具只需在此注册即可
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# 工具定义: 描述每个工具的功能和参数 schema
# LLM 会根据这些定义来决定调用哪个工具
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]


def agent_loop(messages: list):
    """
    智能体核心循环: 与 s01 完全相同, 只是工具调用通过分发映射表路由。

    工作流程:
    1. 调用 LLM API
    2. 追加助手响应
    3. 检查是否需要调用工具
    4. 通过 TOOL_HANDLERS 分发工具调用
    5. 将结果追加到消息历史

    Args:
        messages: 消息历史列表
    """
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                # 通过工具名查找处理函数并执行
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                print(f"> {block.name}: {output[:200]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    """
    主程序入口: 提供交互式命令行界面。

    运行方式:
        python agents/s02_tool_use.py

    支持的命令:
        - 输入任意问题让智能体执行
        - 输入 'q' 或 'exit' 或空行退出程序
    """
    history = []
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
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
