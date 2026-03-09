#!/usr/bin/env python3
"""
s03_todo_write.py - 待办事项写入 (TodoWrite)

模型通过 TodoManager 追踪自己的进度。当模型忘记更新时,
系统会注入一个 nag reminder (唠叨提醒) 强制其更新。

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> | Tools   |
    |  prompt  |      |       |      | + todo  |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                                |
                    +-----------+-----------+
                    | TodoManager state     |
                    | [ ] task A            |
                    | [>] task B <- doing   |
                    | [x] task C            |
                    +-----------------------+
                                |
                    if rounds_since_todo >= 3:
                      inject <reminder>

关键洞察: "智能体可以追踪自己的进度 -- 我也能看到它。"
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

# 系统提示词: 强调使用 todo 工具来规划多步任务
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""


class TodoManager:
    """
    待办事项管理器: 维护结构化的任务状态。

    核心功能:
    1. 存储带状态的任务列表 (pending/in_progress/completed)
    2. 强制同一时间只有一个任务处于 in_progress 状态
    3. 渲染为可读的格式供 LLM 理解

    状态约束:
    - pending: 待处理
    - in_progress: 进行中 (同时只能有一个)
    - completed: 已完成
    """

    def __init__(self):
        """初始化空的待办列表。"""
        self.items = []

    def update(self, items: list) -> str:
        """
        更新待办列表。

        验证规则:
        1. 最多 20 个待办项
        2. 每项必须有 text 字段
        3. status 只能是 pending/in_progress/completed
        4. 同时只能有一个 in_progress

        Args:
            items: 待办项列表, 每项包含 id, text, status

        Returns:
            str: 渲染后的待办列表字符串

        Raises:
            ValueError: 当验证失败时抛出
        """
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")

        validated = []
        in_progress_count = 0

        for i, item in enumerate(items):
            # 提取并验证各字段
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))

            # 验证必填字段
            if not text:
                raise ValueError(f"Item {item_id}: text required")

            # 验证状态值
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")

            # 统计进行中的任务数
            if status == "in_progress":
                in_progress_count += 1

            validated.append({"id": item_id, "text": text, "status": status})

        # 强制: 同时只能有一个任务在进行
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")

        self.items = validated
        return self.render()

    def render(self) -> str:
        """
        将待办列表渲染为可读字符串。

        格式示例:
            [ ] #1: 创建文件
            [>] #2: 编写代码  <- 正在进行
            [x] #3: 测试功能

            (1/3 completed)

        Returns:
            str: 格式化的待办列表
        """
        if not self.items:
            return "No todos."

        lines = []
        for item in self.items:
            # 状态标记: pending=[ ], in_progress=[>], completed=[x]
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")

        # 显示完成进度
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")

        return "\n".join(lines)


# 全局待办管理器实例
TODO = TodoManager()


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

# 工具分发映射表: 将工具名映射到处理函数
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo":       lambda **kw: TODO.update(kw["items"]),  # 新增: 待办事项工具
}

# 工具定义: 描述每个工具的功能和参数 schema
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    # 新增: 待办事项工具 - 让模型能够规划和追踪多步任务
    {"name": "todo", "description": "Update task list. Track progress on multi-step tasks.",
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["id", "text", "status"]}}}, "required": ["items"]}},
]


def agent_loop(messages: list):
    """
    智能体核心循环: 增加了 nag reminder (唠叨提醒) 机制。

    新增功能:
    1. 追踪距离上次调用 todo 工具的轮数
    2. 如果连续 3 轮没有更新待办, 注入提醒消息

    这个机制确保模型在长任务中不会忘记追踪进度。

    Args:
        messages: 消息历史列表
    """
    # 追踪距离上次 todo 更新的轮数
    rounds_since_todo = 0

    while True:
        # 调用 LLM API
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        used_todo = False

        # 执行工具调用
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}: {str(output)[:200]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})

                # 检查是否使用了 todo 工具
                if block.name == "todo":
                    used_todo = True

        # 更新计数器: 如果使用了 todo 则重置, 否则递增
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1

        # Nag reminder: 连续 3 轮未更新待办时注入提醒
        # 这会强制模型关注进度追踪
        if rounds_since_todo >= 3:
            results.insert(0, {"type": "text", "text": "<reminder>Update your todos.</reminder>"})

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    """
    主程序入口: 提供交互式命令行界面。

    运行方式:
        python agents/s03_todo_write.py

    新增特性:
        - 模型会使用 todo 工具规划多步任务
        - 如果模型忘记更新待办, 系统会自动注入提醒
    """
    history = []
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
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
