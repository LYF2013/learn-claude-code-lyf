#!/usr/bin/env python3
"""
s10_team_protocols.py - 团队协议 (Team Protocols)

关机协议和计划审批协议, 两者使用相同的 request_id 关联模式。
在 s09 团队消息系统的基础上构建。

核心概念:
=========
本模块实现两种结构化的请求-响应协议:

1. 关机协议 (Shutdown Protocol):
   - 领导发送关机请求, 队友批准或拒绝
   - 避免直接杀线程导致的数据不一致问题
   - FSM: pending -> approved | rejected

2. 计划审批协议 (Plan Approval Protocol):
   - 队友提交计划, 领导审查批准或拒绝
   - 高风险变更需要先过审再执行
   - FSM: pending -> approved | rejected

协议流程图:
===========

    关机协议 FSM: pending -> approved | rejected

    Lead                              Teammate
    +---------------------+          +---------------------+
    | shutdown_request     |          |                     |
    | {                    | -------> | receives request    |
    |   request_id: abc    |          | decides: approve?   |
    | }                    |          |                     |
    +---------------------+          +---------------------+
                                             |
    +---------------------+          +-------v-------------+
    | shutdown_response    | <------- | shutdown_response   |
    | {                    |          | {                   |
    |   request_id: abc    |          |   request_id: abc   |
    |   approve: true      |          |   approve: true     |
    | }                    |          | }                   |
    +---------------------+          +---------------------+
            |
            v
    status -> "shutdown", thread stops

    计划审批 FSM: pending -> approved | rejected

    Teammate                          Lead
    +---------------------+          +---------------------+
    | plan_approval        |          |                     |
    | submit: {plan:"..."}| -------> | reviews plan text   |
    +---------------------+          | approve/reject?     |
                                     +---------------------+
                                             |
    +---------------------+          +-------v-------------+
    | plan_approval_resp   | <------- | plan_approval       |
    | {approve: true}      |          | review: {req_id,    |
    +---------------------+          |   approve: true}     |
                                     +---------------------+

    追踪器: {request_id: {"target|from": name, "status": "pending|..."}}

关键洞见: "同一个 request_id 关联模式, 两个应用领域。"

相对 s09 的变更:
================
| 组件           | 之前 (s09)       | 之后 (s10)                           |
|----------------|------------------|--------------------------------------|
| Tools          | 9                | 12 (+shutdown_req/resp +plan)        |
| 关机           | 仅自然退出       | 请求-响应握手                        |
| 计划门控       | 无               | 提交/审查与审批                      |
| 关联           | 无               | 每个请求一个 request_id              |
| FSM            | 无               | pending -> approved/rejected         |
"""

import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# =============================================================================
# 全局配置
# =============================================================================

WORKDIR = Path.cwd()  # 工作目录
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))  # Anthropic 客户端
MODEL = os.environ["MODEL_ID"]  # 模型 ID
TEAM_DIR = WORKDIR / ".team"  # 团队配置目录
INBOX_DIR = TEAM_DIR / "inbox"  # 收件箱目录

# 系统提示: 领导角色, 负责管理关机和计划审批协议
SYSTEM = f"You are a team lead at {WORKDIR}. Manage teammates with shutdown and plan approval protocols."

# 有效消息类型集合 (用于消息验证)
VALID_MSG_TYPES = {
    "message",               # 普通消息
    "broadcast",             # 广播消息
    "shutdown_request",      # 关机请求
    "shutdown_response",     # 关机响应
    "plan_approval_response", # 计划审批响应
}

# =============================================================================
# 请求追踪器: 通过 request_id 关联请求和响应
# =============================================================================

shutdown_requests = {}  # 关机请求追踪: {request_id: {"target": name, "status": "pending|..."}}
plan_requests = {}       # 计划请求追踪: {request_id: {"from": name, "plan": text, "status": "pending|..."}}
_tracker_lock = threading.Lock()  # 线程锁, 保护追踪器并发访问


# =============================================================================
# MessageBus: 每个队友一个 JSONL 收件箱
# =============================================================================
class MessageBus:
    """
    消息总线, 管理队友之间的异步消息传递。

    每个队友有一个独立的 JSONL 收件箱文件, 支持发送、读取和广播消息。
    消息按时间戳排序, 读取后清空收件箱 (消费模式)。

    Attributes:
        dir: 收件箱目录路径
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
            extra: 额外字段 (如 request_id, approve 等)

        Returns:
            发送结果消息
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
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        """
        读取并清空指定队友的收件箱。

        Args:
            name: 队友名称

        Returns:
            消息列表 (读取后收件箱被清空)
        """
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        inbox_path.write_text("")  # 清空收件箱
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        """
        向所有队友广播消息 (排除发送者自己)。

        Args:
            sender: 发送者名称
            content: 消息内容
            teammates: 队友名称列表

        Returns:
            广播结果消息
        """
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


# 全局消息总线实例
BUS = MessageBus(INBOX_DIR)


# =============================================================================
# TeammateManager: 管理队友线程 (包含关机 + 计划审批协议)
# =============================================================================
class TeammateManager:
    """
    队友管理器, 负责创建、追踪和管理队友线程。

    相对 s09 的增强:
    - 支持 shutdown_request/shutdown_response 关机握手协议
    - 支持 plan_approval 计划审批协议
    - 队友可以提交计划等待领导批准后再执行

    Attributes:
        dir: 团队配置目录
        config_path: 配置文件路径
        config: 配置字典 (包含 team_name 和 members)
        threads: 线程字典 {name: Thread}
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
        self.threads = {}

    def _load_config(self) -> dict:
        """
        加载团队配置文件。

        Returns:
            配置字典, 如果文件不存在则返回默认配置
        """
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        """
        保存配置到文件。
        """
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        """
        查找指定名称的队友。

        Args:
            name: 队友名称

        Returns:
            队友配置字典, 如果不存在则返回 None
        """
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        """
        启动一个新的队友线程。

        如果队友已存在且状态为 idle 或 shutdown, 则复用;
        否则创建新的队友。

        Args:
            name: 队友名称
            role: 角色描述
            prompt: 初始任务提示

        Returns:
            操作结果消息
        """
        member = self._find_member(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save_config()

        # 创建并启动线程
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
        队友的主循环 (在独立线程中运行)。

        循环逻辑:
        1. 读取收件箱消息
        2. 调用 LLM 生成响应
        3. 执行工具调用 (包括 shutdown_response 和 plan_approval)
        4. 如果批准关机则退出循环

        Args:
            name: 队友名称
            role: 角色描述
            prompt: 初始任务提示
        """
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Submit plans via plan_approval before major work. "
            f"Respond to shutdown_request with shutdown_response."
        )
        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()
        should_exit = False  # 关机标志

        for _ in range(50):  # 最多 50 轮对话
            # 读取收件箱消息
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                messages.append({"role": "user", "content": json.dumps(msg)})

            if should_exit:
                break

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
                    # 如果批准关机, 设置退出标志
                    if block.name == "shutdown_response" and block.input.get("approve"):
                        should_exit = True
            messages.append({"role": "user", "content": results})

        # 更新队友状态
        member = self._find_member(name)
        if member:
            member["status"] = "shutdown" if should_exit else "idle"
            self._save_config()

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        """
        执行工具调用。

        相对 s09 新增:
        - shutdown_response: 响应关机请求
        - plan_approval: 提交计划审批

        Args:
            sender: 发送者 (队友名称)
            tool_name: 工具名称
            args: 工具参数

        Returns:
            工具执行结果
        """
        # 基础工具 (与 s02 相同)
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

        # 关机响应: 队友批准或拒绝关机请求
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            approve = args["approve"]
            with _tracker_lock:
                if req_id in shutdown_requests:
                    shutdown_requests[req_id]["status"] = "approved" if approve else "rejected"
            BUS.send(
                sender, "lead", args.get("reason", ""),
                "shutdown_response", {"request_id": req_id, "approve": approve},
            )
            return f"Shutdown {'approved' if approve else 'rejected'}"

        # 计划审批: 队友提交计划等待领导批准
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())[:8]  # 生成唯一请求 ID
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
            BUS.send(
                sender, "lead", plan_text, "plan_approval_response",
                {"request_id": req_id, "plan": plan_text},
            )
            return f"Plan submitted (request_id={req_id}). Waiting for lead approval."

        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self) -> list:
        """
        返回队友可用的工具定义列表。

        相对 s09 新增:
        - shutdown_response: 响应关机请求
        - plan_approval: 提交计划审批

        Returns:
            工具定义列表
        """
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
            # 新增: 关机响应工具
            {"name": "shutdown_response", "description": "Respond to a shutdown request. Approve to shut down, reject to keep working.",
             "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "reason": {"type": "string"}}, "required": ["request_id", "approve"]}},
            # 新增: 计划审批工具
            {"name": "plan_approval", "description": "Submit a plan for lead approval. Provide plan text.",
             "input_schema": {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]}},
        ]

    def list_all(self) -> str:
        """
        列出所有队友及其状态。

        Returns:
            队友列表的格式化字符串
        """
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        """
        获取所有队友名称列表。

        Returns:
            队友名称列表
        """
        return [m["name"] for m in self.config["members"]]


# 全局队友管理器实例
TEAM = TeammateManager(TEAM_DIR)


# -- Base tool implementations (these base tools are unchanged from s02) --
def _safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
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
    try:
        lines = _safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _run_write(path: str, content: str) -> str:
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = _safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- Lead-specific protocol handlers --
def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send(
        "lead", teammate, "Please shut down gracefully.",
        "shutdown_request", {"request_id": req_id},
    )
    return f"Shutdown request {req_id} sent to '{teammate}' (status: pending)"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    with _tracker_lock:
        req = plan_requests.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    with _tracker_lock:
        req["status"] = "approved" if approve else "rejected"
    BUS.send(
        "lead", req["from"], feedback, "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )
    return f"Plan {req['status']} for '{req['from']}'"


def _check_shutdown_status(request_id: str) -> str:
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}))


# -- Lead tool dispatch (12 tools) --
TOOL_HANDLERS = {
    "bash":              lambda **kw: _run_bash(kw["command"]),
    "read_file":         lambda **kw: _run_read(kw["path"], kw.get("limit")),
    "write_file":        lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file":         lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "spawn_teammate":    lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":    lambda **kw: TEAM.list_all(),
    "send_message":      lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":        lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":         lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
    "shutdown_request":  lambda **kw: handle_shutdown_request(kw["teammate"]),
    "shutdown_response": lambda **kw: _check_shutdown_status(kw.get("request_id", "")),
    "plan_approval":     lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
}

# these base tools are unchanged from s02
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "spawn_teammate", "description": "Spawn a persistent teammate.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "List all teammates.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "send_message", "description": "Send a message to a teammate.",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "Read and drain the lead's inbox.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "broadcast", "description": "Send a message to all teammates.",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
    {"name": "shutdown_request", "description": "Request a teammate to shut down gracefully. Returns a request_id for tracking.",
     "input_schema": {"type": "object", "properties": {"teammate": {"type": "string"}}, "required": ["teammate"]}},
    {"name": "shutdown_response", "description": "Check the status of a shutdown request by request_id.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}}, "required": ["request_id"]}},
    {"name": "plan_approval", "description": "Approve or reject a teammate's plan. Provide request_id + approve + optional feedback.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}}, "required": ["request_id", "approve"]}},
]


def agent_loop(messages: list):
    while True:
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
    history = []
    while True:
        try:
            query = input("\033[36ms10 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
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
