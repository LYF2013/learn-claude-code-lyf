#!/usr/bin/env python3
"""
s11_autonomous_agents.py - 自治智能体 (Autonomous Agents)

空闲循环 + 任务看板轮询 + 自动认领未分配任务 + 上下文压缩后身份重注入。
在 s10 团队协议的基础上构建。

核心概念:
=========
本模块实现队友的自治能力, 队友可以自己扫描任务看板、认领任务、
在空闲时等待新任务, 不需要领导逐个分配。

关键特性:
1. 空闲循环 (Idle Cycle): WORK 和 IDLE 两阶段循环
2. 任务看板轮询: 定期扫描 .tasks/ 目录找未认领的任务
3. 自动认领: 发现未分配任务自动 claim
4. 身份重注入: 上下文压缩后重新注入身份信息

队友生命周期:
=============

    +-------+
    | spawn |  启动
    +---+---+
        |
        v
    +-------+   tool_use    +-------+
    | WORK  | <------------ |  LLM  |   工作阶段: 执行任务
    +---+---+               +-------+
        |
        | stop_reason != tool_use (或调用 idle 工具)
        v
    +--------+
    |  IDLE  |  空闲阶段: 每 5 秒轮询一次, 最多 60 秒
    +---+----+
        |
        +---> 检查收件箱 --> 有消息? --> 恢复 WORK
        |
        +---> 扫描 .tasks/ --> 有未认领任务? --> claim --> 恢复 WORK
        |
        +---> 超时 (60秒) --> 关机

    身份重注入 (上下文压缩后):
    messages = [identity_block, ...remaining...]
    "You are 'coder', role: backend, team: my-team"

关键洞见: "智能体自己找工作。"

相对 s10 的变更:
================
| 组件           | 之前 (s10)       | 之后 (s11)                       |
|----------------|------------------|----------------------------------|
| Tools          | 12               | 14 (+idle, +claim_task)          |
| 自治性         | 领导指派         | 自组织                           |
| 空闲阶段       | 无               | 轮询收件箱 + 任务看板            |
| 任务认领       | 仅手动           | 自动认领未分配任务               |
| 身份           | 系统提示         | + 压缩后重注入                   |
| 超时           | 无               | 60 秒空闲 -> 自动关机            |
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
TASKS_DIR = WORKDIR / ".tasks"  # 任务看板目录

# 空闲轮询配置
POLL_INTERVAL = 5  # 轮询间隔 (秒)
IDLE_TIMEOUT = 60  # 空闲超时 (秒), 超时后自动关机

# 系统提示: 领导角色, 队友是自治的
SYSTEM = f"You are a team lead at {WORKDIR}. Teammates are autonomous -- they find work themselves."

# 有效消息类型集合
VALID_MSG_TYPES = {
    "message",               # 普通消息
    "broadcast",             # 广播消息
    "shutdown_request",      # 关机请求
    "shutdown_response",     # 关机响应
    "plan_approval_response", # 计划审批响应
}

# =============================================================================
# 请求追踪器
# =============================================================================

shutdown_requests = {}  # 关机请求追踪
plan_requests = {}       # 计划请求追踪
_tracker_lock = threading.Lock()  # 追踪器线程锁
_claim_lock = threading.Lock()     # 任务认领线程锁


# =============================================================================
# MessageBus: 每个队友一个 JSONL 收件箱
# =============================================================================
class MessageBus:
    """
    消息总线, 管理队友之间的异步消息传递。

    每个队友有一个独立的 JSONL 收件箱文件, 支持发送、读取和广播消息。
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
            msg_type: 消息类型
            extra: 额外字段

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
            消息列表
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
        向所有队友广播消息。

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
# 任务看板扫描
# =============================================================================

def scan_unclaimed_tasks() -> list:
    """
    扫描任务看板, 找出未认领的任务。

    筛选条件:
    - status == "pending" (待处理)
    - 无 owner (未分配)
    - 无 blockedBy (未被阻塞)

    Returns:
        未认领任务列表
    """
    TASKS_DIR.mkdir(exist_ok=True)
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())
        if (task.get("status") == "pending"
                and not task.get("owner")
                and not task.get("blockedBy")):
            unclaimed.append(task)
    return unclaimed


def claim_task(task_id: int, owner: str) -> str:
    """
    认领任务。

    将任务的 owner 设置为指定队友, status 设为 in_progress。

    Args:
        task_id: 任务 ID
        owner: 认领者名称

    Returns:
        操作结果消息
    """
    with _claim_lock:
        path = TASKS_DIR / f"task_{task_id}.json"
        if not path.exists():
            return f"Error: Task {task_id} not found"
        task = json.loads(path.read_text())
        task["owner"] = owner
        task["status"] = "in_progress"
        path.write_text(json.dumps(task, indent=2))
    return f"Claimed task #{task_id} for {owner}"


# =============================================================================
# 身份重注入 (上下文压缩后)
# =============================================================================

def make_identity_block(name: str, role: str, team_name: str) -> dict:
    """
    创建身份重注入块。

    当上下文被压缩导致消息过短时, 在开头插入此块帮助智能体恢复身份认知。

    Args:
        name: 队友名称
        role: 角色描述
        team_name: 团队名称

    Returns:
        身份消息块
    """
    return {
        "role": "user",
        "content": f"<identity>You are '{name}', role: {role}, team: {team_name}. Continue your work.</identity>",
    }


# =============================================================================
# TeammateManager: 自治队友管理器
# =============================================================================
class TeammateManager:
    """
    自治队友管理器, 支持空闲循环和自动任务认领。

    相对 s10 的增强:
    - WORK 和 IDLE 两阶段循环
    - 空闲时轮询收件箱和任务看板
    - 自动认领未分配任务
    - 上下文压缩后身份重注入
    - 60 秒空闲超时自动关机

    Attributes:
        dir: 团队配置目录
        config_path: 配置文件路径
        config: 配置字典
        threads: 线程字典
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
        """加载团队配置文件。"""
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        """保存配置到文件。"""
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        """查找指定名称的队友。"""
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def _set_status(self, name: str, status: str):
        """
        更新队友状态。

        Args:
            name: 队友名称
            status: 新状态 (working/idle/shutdown)
        """
        member = self._find_member(name)
        if member:
            member["status"] = status
            self._save_config()

    def spawn(self, name: str, role: str, prompt: str) -> str:
        """
        启动一个新的自治队友线程。

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

        thread = threading.Thread(
            target=self._loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _loop(self, name: str, role: str, prompt: str):
        """
        自治队友的主循环 (WORK 和 IDLE 两阶段)。

        WORK 阶段:
        - 执行任务, 调用 LLM
        - 收到 shutdown_request 或调用 idle 工具时进入 IDLE

        IDLE 阶段:
        - 每 5 秒轮询一次收件箱和任务看板
        - 发现新消息或未认领任务时恢复 WORK
        - 60 秒超时后自动关机

        Args:
            name: 队友名称
            role: 角色描述
            prompt: 初始任务提示
        """
        team_name = self.config["team_name"]
        sys_prompt = (
            f"You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. "
            f"Use idle tool when you have no more work. You will auto-claim new tasks."
        )
        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()

        while True:
            # ========== WORK 阶段: 标准 agent 循环 ==========
            for _ in range(50):
                # 读取收件箱消息
                inbox = BUS.read_inbox(name)
                for msg in inbox:
                    # 收到关机请求立即退出
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown")
                        return
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
                    self._set_status(name, "idle")
                    return

                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason != "tool_use":
                    break

                # 执行工具调用
                results = []
                idle_requested = False
                for block in response.content:
                    if block.type == "tool_use":
                        if block.name == "idle":
                            # 队友请求进入空闲阶段
                            idle_requested = True
                            output = "Entering idle phase. Will poll for new tasks."
                        else:
                            output = self._exec(name, block.name, block.input)
                        print(f"  [{name}] {block.name}: {str(output)[:120]}")
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(output),
                        })
                messages.append({"role": "user", "content": results})

                if idle_requested:
                    break

            # ========== IDLE 阶段: 轮询收件箱和任务看板 ==========
            self._set_status(name, "idle")
            resume = False
            polls = IDLE_TIMEOUT // max(POLL_INTERVAL, 1)  # 60s / 5s = 12 次

            for _ in range(polls):
                time.sleep(POLL_INTERVAL)

                # 检查收件箱
                inbox = BUS.read_inbox(name)
                if inbox:
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                        messages.append({"role": "user", "content": json.dumps(msg)})
                    resume = True
                    break

                # 扫描未认领任务
                unclaimed = scan_unclaimed_tasks()
                if unclaimed:
                    task = unclaimed[0]
                    claim_task(task["id"], name)
                    task_prompt = (
                        f"<auto-claimed>Task #{task['id']}: {task['subject']}\n"
                        f"{task.get('description', '')}</auto-claimed>"
                    )
                    # 身份重注入: 上下文过短时插入身份块
                    if len(messages) <= 3:
                        messages.insert(0, make_identity_block(name, role, team_name))
                        messages.insert(1, {"role": "assistant", "content": f"I am {name}. Continuing."})
                    messages.append({"role": "user", "content": task_prompt})
                    messages.append({"role": "assistant", "content": f"Claimed task #{task['id']}. Working on it."})
                    resume = True
                    break

            # 超时未恢复则关机
            if not resume:
                self._set_status(name, "shutdown")
                return
            self._set_status(name, "working")

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        """
        执行工具调用。

        相对 s10 新增:
        - idle: 进入空闲阶段
        - claim_task: 认领任务

        Args:
            sender: 发送者 (队友名称)
            tool_name: 工具名称
            args: 工具参数

        Returns:
            工具执行结果
        """
        # 基础工具
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

        # 协议工具
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            with _tracker_lock:
                if req_id in shutdown_requests:
                    shutdown_requests[req_id]["status"] = "approved" if args["approve"] else "rejected"
            BUS.send(
                sender, "lead", args.get("reason", ""),
                "shutdown_response", {"request_id": req_id, "approve": args["approve"]},
            )
            return f"Shutdown {'approved' if args['approve'] else 'rejected'}"
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())[:8]
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
            BUS.send(
                sender, "lead", plan_text, "plan_approval_response",
                {"request_id": req_id, "plan": plan_text},
            )
            return f"Plan submitted (request_id={req_id}). Waiting for approval."

        # 新增: 任务认领
        if tool_name == "claim_task":
            return claim_task(args["task_id"], sender)

        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self) -> list:
        """
        返回队友可用的工具定义列表 (共 11 个工具)。

        相对 s10 新增:
        - idle: 信号无更多工作, 进入空闲轮询阶段
        - claim_task: 认领任务

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
            {"name": "shutdown_response", "description": "Respond to a shutdown request.",
             "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "reason": {"type": "string"}}, "required": ["request_id", "approve"]}},
            {"name": "plan_approval", "description": "Submit a plan for lead approval.",
             "input_schema": {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]}},
            # 新增: 空闲工具
            {"name": "idle", "description": "Signal that you have no more work. Enters idle polling phase.",
             "input_schema": {"type": "object", "properties": {}}},
            # 新增: 任务认领
            {"name": "claim_task", "description": "Claim a task from the task board by ID.",
             "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
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
        """获取所有队友名称列表。"""
        return [m["name"] for m in self.config["members"]]


# 全局队友管理器实例
TEAM = TeammateManager(TEAM_DIR)


# =============================================================================
# 基础工具实现 (与 s02 相同)
# =============================================================================

def _safe_path(p: str) -> Path:
    """验证路径安全性, 确保不会逃逸工作目录。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
    """执行 shell 命令 (带安全检查和超时限制)。"""
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
    """编辑文件 (替换精确匹配的文本)。"""
    try:
        fp = _safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# 领导专用协议处理器
# =============================================================================

def handle_shutdown_request(teammate: str) -> str:
    """向指定队友发送关机请求。"""
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send(
        "lead", teammate, "Please shut down gracefully.",
        "shutdown_request", {"request_id": req_id},
    )
    return f"Shutdown request {req_id} sent to '{teammate}'"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    """审查队友提交的计划 (批准或拒绝)。"""
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
    """查询关机请求的状态。"""
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}))


# -- Lead tool dispatch (14 tools) --
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
    "idle":              lambda **kw: "Lead does not idle.",
    "claim_task":        lambda **kw: claim_task(kw["task_id"], "lead"),
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
    {"name": "spawn_teammate", "description": "Spawn an autonomous teammate.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "List all teammates.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "send_message", "description": "Send a message to a teammate.",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "Read and drain the lead's inbox.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "broadcast", "description": "Send a message to all teammates.",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
    {"name": "shutdown_request", "description": "Request a teammate to shut down.",
     "input_schema": {"type": "object", "properties": {"teammate": {"type": "string"}}, "required": ["teammate"]}},
    {"name": "shutdown_response", "description": "Check shutdown request status.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}}, "required": ["request_id"]}},
    {"name": "plan_approval", "description": "Approve or reject a teammate's plan.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}}, "required": ["request_id", "approve"]}},
    {"name": "idle", "description": "Enter idle state (for lead -- rarely used).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "claim_task", "description": "Claim a task from the board by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
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
            query = input("\033[36ms11 >> \033[0m")
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
        if query.strip() == "/tasks":
            TASKS_DIR.mkdir(exist_ok=True)
            for f in sorted(TASKS_DIR.glob("task_*.json")):
                t = json.loads(f.read_text())
                marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
                owner = f" @{t['owner']}" if t.get("owner") else ""
                print(f"  {marker} #{t['id']}: {t['subject']}{owner}")
            continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
