#!/usr/bin/env python3
"""
s05_skill_loading.py - 技能加载 (Skills)

两层技能注入机制, 避免系统提示膨胀:

    Layer 1 (低成本): 系统提示中只放技能名称 (~100 tokens/skill)
    Layer 2 (按需加载): 通过 tool_result 加载完整技能内容

    skills/
      pdf/
        SKILL.md          <-- frontmatter (name, description) + body
      code-review/
        SKILL.md

    System prompt:
    +--------------------------------------+
    | You are a coding agent.              |
    | Skills available:                    |
    |   - pdf: Process PDF files...        |  <-- Layer 1: 仅元数据
    |   - code-review: Review code...      |
    +--------------------------------------+

    当模型调用 load_skill("pdf") 时:
    +--------------------------------------+
    | tool_result:                         |
    | <skill>                              |
    |   Full PDF processing instructions   |  <-- Layer 2: 完整内容
    |   Step 1: ...                        |
    |   Step 2: ...                        |
    | </skill>                             |
    +--------------------------------------+

关键洞察: "不要把所有东西都塞进系统提示。按需加载。"
"""

import os
import re
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

# 技能目录: 存放各领域的专业知识文件
SKILLS_DIR = WORKDIR / "skills"


class SkillLoader:
    """
    技能加载器: 扫描并管理技能文件。

    技能文件结构:
        skills/
          pdf/
            SKILL.md       # 包含 YAML frontmatter + 技能内容
          code-review/
            SKILL.md

    YAML frontmatter 格式:
        ---
        name: pdf
        description: Process PDF files
        tags: document, pdf
        ---

        技能的具体内容...

    两层注入机制:
        Layer 1: get_descriptions() - 返回简短描述, 放入系统提示
        Layer 2: get_content() - 返回完整内容, 通过 tool_result 注入
    """

    def __init__(self, skills_dir: Path):
        """
        初始化技能加载器。

        Args:
            skills_dir: 技能目录路径
        """
        self.skills_dir = skills_dir
        self.skills = {}  # 存储所有技能: {name: {meta, body, path}}
        self._load_all()

    def _load_all(self):
        """
        扫描并加载所有技能文件。

        递归查找 skills_dir 下所有 SKILL.md 文件,
        解析其 YAML frontmatter 和正文内容。
        """
        if not self.skills_dir.exists():
            return

        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            text = f.read_text()
            meta, body = self._parse_frontmatter(text)
            # 如果 frontmatter 中没有 name, 使用目录名
            name = meta.get("name", f.parent.name)
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    def _parse_frontmatter(self, text: str) -> tuple:
        """
        解析 YAML frontmatter。

        Frontmatter 格式:
            ---
            name: skill-name
            description: Skill description
            tags: tag1, tag2
            ---

            技能正文内容...

        Args:
            text: SKILL.md 文件的完整内容

        Returns:
            tuple: (meta_dict, body_string)
                   meta_dict 包含 frontmatter 中的键值对
                   body_string 是 frontmatter 之后的正文
        """
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text

        meta = {}
        for line in match.group(1).strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()

        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        """
        Layer 1: 获取技能描述列表, 用于系统提示。

        返回格式:
            - pdf: Process PDF files [document, pdf]
            - code-review: Review code [review]

        Returns:
            str: 格式化的技能描述字符串
        """
        if not self.skills:
            return "(no skills available)"

        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)

        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """
        Layer 2: 获取技能完整内容, 通过 tool_result 注入。

        当模型调用 load_skill 工具时, 返回完整的技能正文,
        包裹在 <skill> 标签中以便模型识别。

        Args:
            name: 技能名称

        Returns:
            str: 包裹在 <skill> 标签中的完整技能内容,
                 或错误信息 (如果技能不存在)
        """
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"

        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"


# 全局技能加载器实例
SKILL_LOADER = SkillLoader(SKILLS_DIR)

# Layer 1: 技能元数据注入到系统提示中
# 这是"低成本"层, 每个技能只占用约 100 tokens
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.

Skills available:
{SKILL_LOADER.get_descriptions()}"""


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

# 工具分发映射表: 包含新增的 load_skill 工具
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),  # 新增: 技能加载
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
    # 新增: 技能加载工具 - 按需加载专业知识
    {"name": "load_skill", "description": "Load specialized knowledge by name.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "Skill name to load"}}, "required": ["name"]}},
]


def agent_loop(messages: list):
    """
    智能体核心循环: 与之前版本相同, 但支持 load_skill 工具。

    新增功能:
        - load_skill 工具: 按需加载技能的完整内容

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
        python agents/s05_skill_loading.py

    新增特性:
        - 系统提示中包含可用技能列表
        - 使用 load_skill 工具按需加载技能完整内容
    """
    history = []
    while True:
        try:
            query = input("\033[36ms05 >> \033[0m")
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
