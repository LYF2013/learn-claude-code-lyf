#!/usr/bin/env python3
"""
s09_agent_teams.py - 智能体团队 (Agent Teams)

持久化的命名智能体, 使用基于文件的 JSONL 收件箱通信。
每个队友在独立线程中运行自己的智能体循环。

    子智能体 (s04):  spawn -> execute -> return summary -> destroyed (一次性)
    队友 (s09):      spawn -> work -> idle -> work -> ... -> shutdown (持久化)

    .team/config.json                   .team/inbox/
    +----------------------------+      +------------------+
    | {"team_name": "default",   |      | alice.jsonl      |
    |  "members": [              |      | bob.jsonl        |
    |    {"name":"alice",        |      | lead.jsonl       |
    |     "role":"coder",        |      +------------------+
    |     "status":"idle"}       |
    |  ]}                        |      send_message("alice", "fix bug"):
    +----------------------------+        open("alice.jsonl", "a").write(msg)

                                        read_inbox("alice"):
    spawn_teammate("alice","coder",...)   msgs = [json.loads(l) for l in ...]
         |                                open("alice.jsonl", "w").close()
         v                                return msgs  # drain
    Thread: alice             Thread: bob
    +------------------+      +------------------+
    | agent_loop       |      | agent_loop       |
    | status: working  |      | status: idle     |
    | ... runs tools   |      | ... waits ...    |
    | status -> idle   |      |                  |
    +------------------+      +------------------+

    5 种消息类型 (已声明, 但并非全部在此实现):
    +-------------------------+-----------------------------------+
    | message                 | 普通文本消息                       |
    | broadcast               | 发送给所有队友                     |
    | shutdown_request        | 请求优雅关闭 (s10)                |
    | shutdown_response       | 批准/拒绝关闭 (s10)               |
    | plan_approval_response  | 批准/拒绝计划 (s10)               |
    +-------------------------+-----------------------------------+

关键洞察: "可以相互交流的队友。"
"""

import json
import os
import subprocess
import threading
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

# 团队配置和收件箱目录
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"

SYSTEM = f"You are a team lead at {WORKDIR}. Spawn teammates and communicate via inboxes."

# 有效的消息类型
VALID_MSG_TYPES = {
    "message",               # 普通消息
    "broadcast",             # 广播消息
    "shutdown_request",      # 关闭请求 (s10)
    "shutdown_response",     # 关闭响应 (s10)
    "plan_approval_response", # 计划审批响应 (s10)
}


class MessageBus:
    """
    消息总线: 基于 JSONL 文件的收件箱系统。

    每个队友都有一个独立的 JSONL 收件箱文件:
    - 发送消息: 向收件人文件追加一行 JSON
    - 读取收件箱: 读取所有消息并清空文件 (drain 模式)

    这种设计确保:
    1. 消息持久化到磁盘, 不会丢失
    2. 简单的追加写入, 无需复杂的消息队列
    3. 原子性读写 (每行一个完整 JSON)
    """

    def __init__(self, inbox_dir: Path):
        """
        初始化消息总线。

        Args:
            inbox_dir: 收件箱目录路径
        """
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        """
        发送消息到指定队友的收件箱。

        Args:
            sender: 发送者名称
            to: 接收者名称
            content: 消息内容
            msg_type: 消息类型 (message/broadcast/shutdown_request 等)
            extra: 额外的消息字段

        Returns:
            str: 发送结果描述
        """
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"

        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg.update(extra)

        # 追加写入到收件人的 JSONL 文件
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n")

        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        """
        读取并清空指定队友的收件箱。

        注意: 这是 drain 模式, 读取后会清空收件箱,
        确保每条消息只被处理一次。

        Args:
            name: 队友名称

        Returns:
            list: 消息列表
        """
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []

        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))

        # 清空收件箱 (drain)
        inbox_path.write_text("")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        """
        向所有队友广播消息。

        Args:
            sender: 发送者名称
            content: 消息内容
            teammates: 队友名称列表

        Returns:
            str: 广播结果描述
        """
        count = 0
        for name in teammates:
            if name != sender:  # 不发送给自己
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


# 全局消息总线实例
BUS = MessageBus(INBOX_DIR)


class TeammateManager:
    """
    队友管理器: 管理持久化的命名智能体。

    核心功能:
    1. 维护团队名册 (config.json)
    2. 启动队友线程 (每个队友运行独立的 agent loop)
    3. 追踪队友状态 (idle/working/shutdown)

    队友生命周期:
    - spawn: 创建队友并启动线程
    - working: 队友正在执行任务
    - idle: 队友空闲, 可以接受新任务
    - shutdown: 队友正在关闭

    配置文件结构 (config.json):
    {
        "team_name": "default",
        "members": [
            {"name": "alice", "role": "coder", "status": "idle"},
            {"name": "bob", "role": "tester", "status": "working"}
        ]
    }
    """

    def __init__(self, team_dir: Path):
        """
        初始化队友管理器。

        Args:
            team_dir: 团队配置目录路径
        """
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}  # 队友名称 -> 线程对象

    def _load_config(self) -> dict:
        """加载团队配置文件。"""
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        """保存团队配置到文件。"""
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        """查找指定名称的队友配置。"""
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        """
        启动一个队友。

        如果队友已存在且处于 idle 状态, 则重新激活。
        如果队友不存在, 则创建新的队友。

        Args:
            name: 队友名称 (唯一标识)
            role: 队友角色 (如 coder, tester, reviewer)
            prompt: 初始任务提示

        Returns:
            str: 操作结果描述
        """
        member = self._find_member(name)

        if member:
            # 队友已存在, 检查状态
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            # 创建新队友
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)

        self._save_config()

        # 在独立线程中启动队友的 agent loop
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()

        return f"Spawned '{name}' (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        """
        队友的智能体循环 (在独立线程中运行)。

        每轮循环:
        1. 检查收件箱, 将新消息注入上下文
        2. 调用 LLM 获取响应
        3. 执行工具调用
        4. 重复直到任务完成或达到最大轮数

        Args:
            name: 队友名称
            role: 队友角色
            prompt: 初始任务提示
        """
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Use send_message to communicate. Complete your task."
        )
        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()

        # 最多 50 轮交互
        for _ in range(50):
            # 检查收件箱并注入消息
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                messages.append({"role": "user", "content": json.dumps(msg)})

            try:
                response = client.messages.create(
                    model=MODEL,
                    system=sys_prompt,
                    messages=messages,
                    tools=tools,
                    max_tokens=8000,
                )
            except Exception:
                break

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break

            # 执行工具调用
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    output = self._exec(name, block.name, block.input)
                    print(f"  [{name}] {block.name}: {str(output)[:120]}")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    })

            messages.append({"role": "user", "content": results})

        # 任务完成, 更新状态为 idle
        member = self._find_member(name)
        if member and member["status"] != "shutdown":
            member["status"] = "idle"
            self._save_config()

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        """
        执行队友的工具调用。

        Args:
            sender: 调用者 (队友名称)
            tool_name: 工具名称
            args: 工具参数

        Returns:
            str: 工具执行结果
        """
        if tool_name == "bash":
            return _run_bash(args["command"])
        if tool_name == "read_file":
            return _run_read(args["path"])
        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message":
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)
        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self) -> list:
        """返回队友可用的工具列表。"""
        return [
            {"name": "bash", "description": "Run a shell command.",
             "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
            {"name": "read_file", "description": "Read file contents.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            {"name": "write_file", "description": "Write content to file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            {"name": "edit_file", "description": "Replace exact text in file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
            {"name": "send_message", "description": "Send message to a teammate.",
             "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
            {"name": "read_inbox", "description": "Read and drain your inbox.",
             "input_schema": {"type": "object", "properties": {}}},
        ]

    def list_all(self) -> str:
        """列出所有队友及其状态。"""
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        """返回所有队友名称列表。"""
        return [m["name"] for m in self.config["members"]]


# 全局队友管理器实例
TEAM = TeammateManager(TEAM_DIR)


# ============== 基础工具实现函数 ==============

def _safe_path(p: str) -> Path:
    """安全路径解析: 确保路径不会逃逸出工作目录。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
    """执行 bash 命令并返回结果。"""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def _run_read(path: str, limit: int = None) -> str:
    """读取文件内容。"""
    try:
        lines = _safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _run_write(path: str, content: str) -> str:
    """写入文件内容。"""
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
    """编辑文件: 替换首次匹配的文本。"""
    try:
        fp = _safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ============== 领导者工具分发 ==============

# 工具分发映射表 (领导者有 9 个工具)
TOOL_HANDLERS = {
    # 基础工具
    "bash":            lambda **kw: _run_bash(kw["command"]),
    "read_file":       lambda **kw: _run_read(kw["path"], kw.get("limit")),
    "write_file":      lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file":       lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    # 团队管理工具
    "spawn_teammate":  lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":  lambda **kw: TEAM.list_all(),
    # 通信工具
    "send_message":    lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":      lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":       lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
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
    # 团队管理工具
    {"name": "spawn_teammate", "description": "Spawn a persistent teammate that runs in its own thread.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "List all teammates with name, role, status.",
     "input_schema": {"type": "object", "properties": {}}},
    # 通信工具
    {"name": "send_message", "description": "Send a message to a teammate's inbox.",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "Read and drain the lead's inbox.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "broadcast", "description": "Send a message to all teammates.",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
]


def agent_loop(messages: list):
    """
    领导者智能体的核心循环。

    与之前版本不同, 领导者在每次 LLM 调用前会检查自己的收件箱,
    将收到的消息注入到上下文中。

    工作流程:
    1. 检查收件箱, 将新消息注入上下文
    2. 调用 LLM 获取响应
    3. 执行工具调用
    4. 重复

    Args:
        messages: 消息历史列表
    """
    while True:
        # 检查领导者收件箱并注入消息
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
            })
            messages.append({
                "role": "assistant",
                "content": "Noted inbox messages.",
            })

        # 调用 LLM API
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        # 执行工具调用
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}: {str(output)[:200]}")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output),
                })

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    """
    主程序入口: 提供交互式命令行界面。

    运行方式:
        python agents/s09_agent_teams.py

    新增特性:
        - 支持创建持久化的队友 (spawn_teammate)
        - 队友在独立线程中运行
        - 通过 JSONL 收件箱进行通信
        - 领导者可以管理和与队友交互

    特殊命令:
        /team  - 查看团队名册和状态
        /inbox - 检查领导者的收件箱
    """
    history = []
    while True:
        try:
            query = input("\033[36ms09 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        # 特殊命令: 查看团队状态
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        # 特殊命令: 检查收件箱
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
