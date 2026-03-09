#!/usr/bin/env python3
"""
s06_context_compact.py - 上下文压缩 (Compact)

三层压缩流水线, 让智能体可以无限期工作:

    每一轮:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (静默执行, 每轮都运行)
      将超过 3 轮的 tool_result 内容
      替换为 "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > 50000?]
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  保存完整对话到 .transcripts/
                  让 LLM 摘要对话内容
                  用 [summary] 替换所有消息
                        |
                        v
                [Layer 3: compact tool]
                  模型调用 compact -> 立即压缩
                  与 auto 相同, 手动触发

关键洞察: "智能体可以策略性地遗忘, 从而无限工作。"
"""

import json
import os
import subprocess
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

# 压缩阈值: 当 token 估计超过此值时触发 auto_compact
THRESHOLD = 50000
# 历史记录保存目录
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
# 保留最近的 N 个 tool_result 不压缩
KEEP_RECENT = 3


def estimate_tokens(messages: list) -> int:
    """
    粗略估算消息的 token 数量。

    使用简单的启发式方法: 约 4 个字符 = 1 个 token。
    这不是精确计算, 但足够用于触发压缩判断。

    Args:
        messages: 消息历史列表

    Returns:
        int: 估算的 token 数量
    """
    return len(str(messages)) // 4


def micro_compact(messages: list) -> list:
    """
    Layer 1: 微压缩 - 将旧的 tool_result 替换为占位符。

    这是轻量级压缩, 每轮都静默执行:
    1. 收集所有 tool_result 条目
    2. 只保留最近的 KEEP_RECENT 个
    3. 将更早的 tool_result 内容替换为简短占位符

    这样可以显著减少上下文长度, 同时保留必要的信息:
    - 知道使用过什么工具
    - 不保留冗长的输出内容

    Args:
        messages: 消息历史列表 (会被原地修改)

    Returns:
        list: 修改后的消息列表
    """
    # 收集所有 tool_result 条目: (消息索引, 部分索引, tool_result 字典)
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part_idx, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))

    # 如果 tool_result 数量不超过保留数量, 无需压缩
    if len(tool_results) <= KEEP_RECENT:
        return messages

    # 建立 tool_use_id -> tool_name 的映射
    # 用于在压缩时显示工具名称
    tool_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "type") and block.type == "tool_use":
                        tool_name_map[block.id] = block.name

    # 压缩旧的 tool_result (保留最近的 KEEP_RECENT 个)
    to_clear = tool_results[:-KEEP_RECENT]
    for _, _, result in to_clear:
        # 只压缩内容较长的结果
        if isinstance(result.get("content"), str) and len(result["content"]) > 100:
            tool_id = result.get("tool_use_id", "")
            tool_name = tool_name_map.get(tool_id, "unknown")
            result["content"] = f"[Previous: used {tool_name}]"

    return messages


def auto_compact(messages: list) -> list:
    """
    Layer 2 & 3: 自动/手动压缩 - 完整摘要并替换消息历史。

    这是重量级压缩, 当 token 超过阈值或模型主动调用 compact 时触发:
    1. 将完整对话历史保存到 .transcripts/ 目录 (便于恢复)
    2. 调用 LLM 生成对话摘要
    3. 用摘要替换所有消息历史

    摘要包含:
    - 已完成的工作
    - 当前状态
    - 关键决策

    Args:
        messages: 消息历史列表

    Returns:
        list: 压缩后的新消息列表 (只包含摘要)
    """
    # 保存完整对话历史到磁盘
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")

    # 让 LLM 生成摘要
    conversation_text = json.dumps(messages, default=str)[:80000]
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity. Include: "
            "1) What was accomplished, 2) Current state, 3) Key decisions made. "
            "Be concise but preserve critical details.\n\n" + conversation_text}],
        max_tokens=2000,
    )
    summary = response.content[0].text

    # 用压缩后的摘要替换所有消息
    return [
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
        {"role": "assistant", "content": "Understood. I have the context from the summary. Continuing."},
    ]


# ============== 工具实现函数 ==============

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


# ============== 工具注册 ==============

# 工具分发映射表: 包含新增的 compact 工具
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "compact":    lambda **kw: "Manual compression requested.",  # 新增: 手动压缩
}

# 工具定义列表
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    # 新增: 手动压缩工具 - 让模型可以主动触发上下文压缩
    {"name": "compact", "description": "Trigger manual conversation compression.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string", "description": "What to preserve in the summary"}}}},
]


def agent_loop(messages: list):
    """
    智能体核心循环: 整合三层压缩机制。

    三层压缩策略:
    1. Layer 1 (micro_compact): 每轮静默执行, 压缩旧的 tool_result
    2. Layer 2 (auto_compact): token 超过阈值时自动触发
    3. Layer 3 (manual compact): 模型主动调用 compact 工具时触发

    压缩顺序:
    - 每次 LLM 调用前: micro_compact
    - token 超过阈值时: auto_compact
    - 模型调用 compact 工具后: auto_compact

    Args:
        messages: 消息历史列表 (会被原地修改)
    """
    while True:
        # Layer 1: 微压缩 - 每次 LLM 调用前执行
        micro_compact(messages)

        # Layer 2: 自动压缩 - token 超过阈值时触发
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)

        # 调用 LLM API
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        manual_compact = False

        # 执行工具调用
        for block in response.content:
            if block.type == "tool_use":
                # 特殊处理 compact 工具: 标记需要手动压缩
                if block.name == "compact":
                    manual_compact = True
                    output = "Compressing..."
                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    try:
                        output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                    except Exception as e:
                        output = f"Error: {e}"

                print(f"> {block.name}: {str(output)[:200]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})

        messages.append({"role": "user", "content": results})

        # Layer 3: 手动压缩 - 模型调用 compact 工具后触发
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)


if __name__ == "__main__":
    """
    主程序入口: 提供交互式命令行界面。

    运行方式:
        python agents/s06_context_compact.py

    新增特性:
        - 三层上下文压缩机制
        - micro_compact: 每轮静默压缩旧的 tool_result
        - auto_compact: token 超过阈值时自动摘要
        - compact 工具: 模型可主动触发压缩
        - 完整对话历史保存在 .transcripts/ 目录
    """
    history = []
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
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
