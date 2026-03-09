#!/usr/bin/env python3
"""
s07_task_system.py - 任务系统 (Tasks)

任务持久化为 JSON 文件存储在 .tasks/ 目录, 因此可以在上下文压缩后存活。
每个任务都有依赖图 (blockedBy/blocks)。

    .tasks/
      task_1.json  {"id":1, "subject":"...", "status":"completed", ...}
      task_2.json  {"id":2, "blockedBy":[1], "status":"pending", ...}
      task_3.json  {"id":3, "blockedBy":[2], "blocks":[], ...}

    依赖解析:
    +----------+     +----------+     +----------+
    | task 1   | --> | task 2   | --> | task 3   |
    | complete |     | blocked  |     | blocked  |
    +----------+     +----------+     +----------+
         |                ^
         +--- 完成任务 1 会将其从任务 2 的 blockedBy 中移除

关键洞察: "能存活压缩的状态 -- 因为它在对话之外。"
"""

import json
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

# 任务存储目录: 任务以 JSON 文件形式持久化
TASKS_DIR = WORKDIR / ".tasks"

SYSTEM = f"You are a coding agent at {WORKDIR}. Use task tools to plan and track work."


class TaskManager:
    """
    任务管理器: 实现任务的 CRUD 操作和依赖图管理。

    每个任务是一个 JSON 文件, 包含:
    - id: 唯一标识符
    - subject: 任务标题
    - description: 任务描述
    - status: 状态 (pending/in_progress/completed)
    - blockedBy: 前置依赖列表 (必须先完成的任务 ID)
    - blocks: 后置依赖列表 (此任务完成后被解锁的任务 ID)
    - owner: 任务所有者 (用于多 agent 协作)

    核心功能:
    1. 创建任务并自动分配 ID
    2. 更新任务状态和依赖关系
    3. 任务完成时自动解锁后续任务
    4. 列出所有任务及其状态

    持久化:
    - 任务存储在 .tasks/ 目录下的 JSON 文件中
    - 可以在上下文压缩和程序重启后存活
    """

    def __init__(self, tasks_dir: Path):
        """
        初始化任务管理器。

        Args:
            tasks_dir: 任务存储目录路径
        """
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        """
        获取当前最大的任务 ID。

        Returns:
            int: 最大的任务 ID, 如果没有任务则返回 0
        """
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0

    def _load(self, task_id: int) -> dict:
        """
        加载指定任务的数据。

        Args:
            task_id: 任务 ID

        Returns:
            dict: 任务数据字典

        Raises:
            ValueError: 如果任务不存在
        """
        path = self.dir / f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        """
        保存任务数据到 JSON 文件。

        Args:
            task: 任务数据字典
        """
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(json.dumps(task, indent=2))

    def create(self, subject: str, description: str = "") -> str:
        """
        创建新任务。

        Args:
            subject: 任务标题
            description: 任务描述 (可选)

        Returns:
            str: JSON 格式的任务数据
        """
        task = {
            "id": self._next_id, "subject": subject, "description": description,
            "status": "pending", "blockedBy": [], "blocks": [], "owner": "",
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2)

    def get(self, task_id: int) -> str:
        """
        获取指定任务的详细信息。

        Args:
            task_id: 任务 ID

        Returns:
            str: JSON 格式的任务数据
        """
        return json.dumps(self._load(task_id), indent=2)

    def update(self, task_id: int, status: str = None,
               add_blocked_by: list = None, add_blocks: list = None) -> str:
        """
        更新任务的状态和依赖关系。

        状态转换:
        - pending: 待处理
        - in_progress: 进行中
        - completed: 已完成 (会自动解锁后续任务)

        依赖关系:
        - add_blocked_by: 添加前置依赖 (此任务需要等待这些任务完成)
        - add_blocks: 添加后置依赖 (此任务完成后会解锁这些任务)

        Args:
            task_id: 任务 ID
            status: 新状态 (可选)
            add_blocked_by: 添加的前置依赖列表 (可选)
            add_blocks: 添加的后置依赖列表 (可选)

        Returns:
            str: JSON 格式的更新后任务数据
        """
        task = self._load(task_id)

        # 更新状态
        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            # 当任务完成时, 将其从所有其他任务的 blockedBy 中移除
            # 这会"解锁"依赖于此任务的其他任务
            if status == "completed":
                self._clear_dependency(task_id)

        # 添加前置依赖 (此任务需要等待的任务)
        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))

        # 添加后置依赖 (此任务完成后会解锁的任务)
        if add_blocks:
            task["blocks"] = list(set(task["blocks"] + add_blocks))
            # 双向更新: 同时更新被阻塞任务的 blockedBy 列表
            for blocked_id in add_blocks:
                try:
                    blocked = self._load(blocked_id)
                    if task_id not in blocked["blockedBy"]:
                        blocked["blockedBy"].append(task_id)
                        self._save(blocked)
                except ValueError:
                    pass

        self._save(task)
        return json.dumps(task, indent=2)

    def _clear_dependency(self, completed_id: int):
        """
        清除依赖关系: 将已完成的任务 ID 从所有其他任务的 blockedBy 中移除。

        当任务完成时调用此方法, 自动解锁等待该任务的其他任务。

        Args:
            completed_id: 已完成的任务 ID
        """
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)

    def list_all(self) -> str:
        """
        列出所有任务及其状态摘要。

        显示格式:
            [ ] #1: 任务标题 (blocked by: [2, 3])
            [>] #2: 进行中的任务
            [x] #3: 已完成的任务

        状态标记:
            [ ] - pending (待处理)
            [>] - in_progress (进行中)
            [x] - completed (已完成)

        Returns:
            str: 格式化的任务列表
        """
        tasks = []
        for f in sorted(self.dir.glob("task_*.json")):
            tasks.append(json.loads(f.read_text()))

        if not tasks:
            return "No tasks."

        lines = []
        for t in tasks:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{blocked}")

        return "\n".join(lines)


# 全局任务管理器实例
TASKS = TaskManager(TASKS_DIR)


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
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ============== 工具注册 ==============

# 工具分发映射表: 包含基础工具和任务管理工具
TOOL_HANDLERS = {
    "bash":        lambda **kw: run_bash(kw["command"]),
    "read_file":   lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":  lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":   lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    # 任务管理工具
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("addBlockedBy"), kw.get("addBlocks")),
    "task_list":   lambda **kw: TASKS.list_all(),
    "task_get":    lambda **kw: TASKS.get(kw["task_id"]),
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
    # 任务管理工具
    {"name": "task_create", "description": "Create a new task.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}},
    {"name": "task_update", "description": "Update a task's status or dependencies.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "addBlockedBy": {"type": "array", "items": {"type": "integer"}}, "addBlocks": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks with status summary.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "task_get", "description": "Get full details of a task by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
]


def agent_loop(messages: list):
    """
    智能体核心循环: 与之前版本相同, 但支持任务管理工具。

    新增功能:
        - task_create: 创建新任务
        - task_update: 更新任务状态和依赖关系
        - task_list: 列出所有任务
        - task_get: 获取任务详情

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
        python agents/s07_task_system.py

    新增特性:
        - 任务以 JSON 文件形式持久化到 .tasks/ 目录
        - 支持任务之间的依赖关系 (blockedBy/blocks)
        - 任务完成时自动解锁后续任务
        - 可以在上下文压缩和程序重启后恢复状态
    """
    history = []
    while True:
        try:
            query = input("\033[36ms07 >> \033[0m")
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
