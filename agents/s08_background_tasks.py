#!/usr/bin/env python3
"""
s08_background_tasks.py - 后台任务 (Background Tasks)

在后台线程中运行命令。通知队列在每次 LLM 调用前排空,
将完成的结果注入到对话中。

    主线程                      后台线程
    +-----------------+        +-----------------+
    | agent loop      |        | task executes   |
    | ...             |        | ...             |
    | [LLM call] <---+------- | enqueue(result) |
    |  ^drain queue   |        +-----------------+
    +-----------------+

    时间线:
    Agent ----[spawn A]----[spawn B]----[other work]----
                 |              |
                 v              v
              [A runs]      [B runs]        (并行执行)
                 |              |
                 +-- notification queue --> [结果注入]

关键洞察: "发射后不管 -- agent 不需要阻塞等待命令执行。"
"""

import os
import subprocess
import threading
import uuid
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use background_run for long-running commands."


class BackgroundManager:
    """
    后台任务管理器: 在后台线程执行命令并管理通知队列。

    核心功能:
    1. 启动后台线程执行长时间运行的命令
    2. 追踪所有后台任务的状态
    3. 通过通知队列在命令完成后通知主线程

    线程安全:
    - 使用 threading.Lock 保护通知队列
    - 使用守护线程 (daemon=True) 确保程序可以正常退出

    使用流程:
    1. 调用 run(command) 启动后台任务, 立即返回 task_id
    2. 后台线程执行命令, 完成后将结果加入通知队列
    3. 主线程在每次 LLM 调用前调用 drain_notifications() 获取完成的通知
    """

    def __init__(self):
        """初始化后台任务管理器。"""
        self.tasks = {}  # task_id -> {status, result, command}
        self._notification_queue = []  # 已完成任务的通知列表
        self._lock = threading.Lock()  # 保护通知队列的锁

    def run(self, command: str) -> str:
        """
        启动后台线程执行命令, 立即返回 task_id。

        这是"发射后不管"模式:
        - 命令在后台线程中执行
        - 主线程可以继续处理其他工作
        - 命令完成后通过通知队列通知

        Args:
            command: 要执行的 shell 命令

        Returns:
            str: 包含 task_id 的状态消息
        """
        task_id = str(uuid.uuid4())[:8]  # 生成短 ID
        self.tasks[task_id] = {"status": "running", "result": None, "command": command}

        # 启动守护线程执行命令
        thread = threading.Thread(
            target=self._execute, args=(task_id, command), daemon=True
        )
        thread.start()
        return f"Background task {task_id} started: {command[:80]}"

    def _execute(self, task_id: str, command: str):
        """
        后台线程的目标函数: 执行子进程并捕获输出。

        执行流程:
        1. 运行子进程 (最长 300 秒超时)
        2. 捕获 stdout 和 stderr
        3. 更新任务状态
        4. 将完成通知加入队列

        Args:
            task_id: 任务 ID
            command: 要执行的 shell 命令
        """
        try:
            r = subprocess.run(
                command, shell=True, cwd=WORKDIR,
                capture_output=True, text=True, timeout=300
            )
            output = (r.stdout + r.stderr).strip()[:50000]
            status = "completed"
        except subprocess.TimeoutExpired:
            output = "Error: Timeout (300s)"
            status = "timeout"
        except Exception as e:
            output = f"Error: {e}"
            status = "error"

        # 更新任务状态
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = output or "(no output)"

        # 将完成通知加入队列 (线程安全)
        with self._lock:
            self._notification_queue.append({
                "task_id": task_id,
                "status": status,
                "command": command[:80],
                "result": (output or "(no output)")[:500],  # 通知中只显示前 500 字符
            })

    def check(self, task_id: str = None) -> str:
        """
        检查任务状态。

        Args:
            task_id: 可选的任务 ID。如果提供, 返回该任务的详细信息;
                     如果不提供, 返回所有任务的列表。

        Returns:
            str: 任务状态信息
        """
        if task_id:
            t = self.tasks.get(task_id)
            if not t:
                return f"Error: Unknown task {task_id}"
            return f"[{t['status']}] {t['command'][:60]}\n{t.get('result') or '(running)'}"

        # 列出所有任务
        lines = []
        for tid, t in self.tasks.items():
            lines.append(f"{tid}: [{t['status']}] {t['command'][:60]}")
        return "\n".join(lines) if lines else "No background tasks."

    def drain_notifications(self) -> list:
        """
        排空通知队列: 返回并清空所有待处理的通知。

        这个方法在每次 LLM 调用前被调用, 将后台任务完成的通知
        注入到对话中, 让模型知道哪些命令已经完成。

        Returns:
            list: 通知列表, 每个通知包含 task_id, status, command, result
        """
        with self._lock:
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        return notifs


# 全局后台任务管理器实例
BG = BackgroundManager()


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
    执行 bash 命令并返回结果 (阻塞式)。

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
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ============== 工具注册 ==============

# 工具分发映射表: 包含阻塞式和后台式命令工具
TOOL_HANDLERS = {
    "bash":             lambda **kw: run_bash(kw["command"]),
    "read_file":        lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":       lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":        lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    # 后台任务工具
    "background_run":   lambda **kw: BG.run(kw["command"]),  # 启动后台任务
    "check_background": lambda **kw: BG.check(kw.get("task_id")),  # 检查任务状态
}

# 工具定义列表
TOOLS = [
    {"name": "bash", "description": "Run a shell command (blocking).",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    # 后台任务工具
    {"name": "background_run", "description": "Run command in background thread. Returns task_id immediately.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "check_background", "description": "Check background task status. Omit task_id to list all.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}}},
]


def agent_loop(messages: list):
    """
    智能体核心循环: 整合后台任务通知机制。

    新增功能:
        1. 每次 LLM 调用前排空后台任务通知队列
        2. 将完成的后台任务结果注入到对话中
        3. 支持阻塞式 (bash) 和非阻塞式 (background_run) 命令

    通知注入流程:
        1. 检查通知队列
        2. 如果有待处理通知, 将其格式化为 <background-results> 消息
        3. 添加用户消息 (通知) 和助手确认
        4. 继续正常的 LLM 调用

    Args:
        messages: 消息历史列表
    """
    while True:
        # 在 LLM 调用前排空后台通知队列
        notifs = BG.drain_notifications()
        if notifs and messages:
            notif_text = "\n".join(
                f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifs
            )
            # 将通知注入到对话中
            messages.append({"role": "user", "content": f"<background-results>\n{notif_text}\n</background-results>"})
            messages.append({"role": "assistant", "content": "Noted background results."})

        # 调用 LLM API
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
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}: {str(output)[:200]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    """
    主程序入口: 提供交互式命令行界面。

    运行方式:
        python agents/s08_background_tasks.py

    新增特性:
        - 支持后台执行长时间运行的命令
        - 后台任务完成后自动通知
        - 可以同时运行多个后台任务
        - 阻塞式命令 (bash) 和非阻塞式命令 (background_run) 并存
    """
    history = []
    while True:
        try:
            query = input("\033[36ms08 >> \033[0m")
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
