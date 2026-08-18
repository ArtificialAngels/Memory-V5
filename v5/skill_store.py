"""Skill 记忆存储层 —— Markdown 文件即记忆 (借鉴 memU 的磁盘即接口设计).

存储: ``data/v5/skills/<name>.md``, front-matter 只有 name + description
(description 用于检索索引), 正文是可复用工作流.

设计原则 (对应 memU 调研结论):
- 文件系统是唯一真源, 不写 DB 双份 → 可 diff / 可 git / 哥哥可直接审阅.
- 检索走渐进形状: 先返回窄命中 (name/description 索引), 全文按需读取.
- 零外部依赖: 不依赖嵌入服务, 关键词 + 加权评分, 技能数量少 (几十个)
  时扫描足够快.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path

# 与 store.py 同一数据根 (禁硬编码盘符, 由包路径推导)
MEM_ROOT = Path(__file__).resolve().parent
SKILLS_DIR = MEM_ROOT / "data" / "v5" / "skills"

# front-matter 字段
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$", re.ASCII)

# 检索权重: name 命中 > description 命中 > 正文命中
_W_NAME = 3.0
_W_DESC = 2.0
_W_BODY = 1.0
# 正文命中分数上限, 防止长文档刷分
_BODY_CAP = 3.0

_FRONT_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)


def skills_dir() -> Path:
    """技能目录 (延迟创建)."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    return SKILLS_DIR


def _validate_name(name: str) -> str:
    name = (name or "").strip()
    if not _NAME_RE.match(name):
        raise ValueError(
            f"skill name 必须是 kebab-case (小写字母/数字/连字符, ≤64): {name!r}"
        )
    return name


def skill_path(name: str) -> Path:
    """校验并返回技能文件路径."""
    return skills_dir() / f"{_validate_name(name)}.md"


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """解析 front-matter, 返回 (meta, body). 无 front-matter 时返回空 meta."""
    m = _FRONT_RE.match(text)
    if not m:
        return {}, text.strip()
    meta: dict = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, text[m.end():].strip()


def _render(name: str, description: str, content: str) -> str:
    """渲染技能文件: front-matter + 正文."""
    body = (content or "").strip()
    head = f"---\nname: {name}\ndescription: {description}\n---\n"
    return head + ("\n" + body if body else "")


def write_skill(name: str, description: str, content: str) -> dict:
    """创建或更新技能文件 (原子写: 先写临时文件再 replace).

    - ``name``: kebab-case 文件名.
    - ``description``: 一句话摘要, 检索索引文本.
    - ``content``: 技能正文 (可复用工作流).
    """
    name = _validate_name(name)
    description = (description or "").strip()
    content = (content or "").strip()
    if not description:
        raise ValueError("skill description 不能为空 (它是检索索引)")
    if not content:
        raise ValueError("skill content 不能为空")

    path = skill_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    payload = _render(name, description, content)

    # 原子写: 同目录临时文件 + os.replace (Windows 下原子替换)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=f".{name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    return {
        "ok": True,
        "name": name,
        "description": description,
        "path": str(path),
        "created": not existed,
        "updated": time.time(),
    }


def _scan() -> list[tuple[Path, dict, str]]:
    """扫描技能目录, 返回 [(path, meta, body)] (损坏文件跳过)."""
    out: list[tuple[Path, dict, str]] = []
    if not SKILLS_DIR.is_dir():
        return out
    for p in sorted(SKILLS_DIR.glob("*.md")):
        if p.name.startswith("."):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        meta, body = _split_frontmatter(text)
        if not meta.get("name") or not meta.get("description"):
            continue  # 非技能文件 (或损坏) 静默跳过
        out.append((p, meta, body))
    return out


def list_skills() -> list[dict]:
    """列出全部技能: name / description / path / updated / body_len."""
    out = []
    for p, meta, body in _scan():
        try:
            updated = p.stat().st_mtime
        except OSError:
            updated = 0.0
        out.append({
            "name": meta["name"],
            "description": meta["description"],
            "path": str(p),
            "updated": updated,
            "body_len": len(body),
        })
    return out


def get_skill(name: str) -> dict | None:
    """读取单个技能全文 (渐进形状的"宽"层: 需要时再打开)."""
    try:
        path = skill_path(name)
    except ValueError:
        return None
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    meta, body = _split_frontmatter(text)
    if not meta.get("name"):
        return None
    return {
        "name": meta["name"],
        "description": meta.get("description", ""),
        "content": body,
        "path": str(path),
        "updated": path.stat().st_mtime,
    }


def remove_skill(name: str) -> bool:
    """删除技能文件. 不存在时返回 False (幂等)."""
    try:
        path = skill_path(name)
    except ValueError:
        return False
    if not path.is_file():
        return False
    path.unlink()
    return True


_CJK = re.compile(r"[\u4e00-\u9fff]+")


def _tokens(query: str) -> list[str]:
    """查询切词: 按空白/标点切; ASCII 词原样 (≥2 字符), 中文串拆 2-gram.

    中英混排 (如 "SOUL注入") 必须拆开 —— "soul" 和 "注入" 是独立检索单元,
    不拆则整串子串匹配几乎必然落空 (正文写的是 "SOUL.md" 和 "注入").
    """
    tokens: list[str] = []
    for part in re.split(r"[\s,，。、;；:：!！?？/\\|()\[\]{}]+", query.strip().lower()):
        if not part:
            continue
        # 纯 ASCII 词: 原样保留
        if re.fullmatch(r"[a-z0-9][a-z0-9-]*", part):
            if len(part) >= 2:
                tokens.append(part)
            continue
        # 含中文: 提取每个连续中文串做 2-gram, ASCII 段按单词拆
        for seg in re.split(r"([\u4e00-\u9fff]+)", part):
            if not seg:
                continue
            if _CJK.fullmatch(seg):
                if len(seg) >= 2:
                    tokens.append(seg)
                if len(seg) >= 3:
                    tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
            else:
                for w in re.findall(r"[a-z0-9]+", seg):
                    if len(w) >= 2:
                        tokens.append(w)
    # 去重保序
    return list(dict.fromkeys(tokens))


def search_skills(query: str, top_k: int = 5) -> list[dict]:
    """渐进形状检索: 只返回窄命中 (name/description/score), 不含全文.

    打分规则:
      - name 精确/包含命中: 强信号
      - description 命中: 索引文本命中
      - 正文命中: 弱信号, 分数封顶 (防长文档刷分)

    返回 [{name, description, path, score, updated}], 按 score 降序.
    全文由调用方按需 get_skill 读取 —— 这正是 memU progressive_retrieve
    的 "给 agent 位置 + 摘要, 而不是全文" 原则.
    """
    query = (query or "").strip()
    if not query:
        return []
    tokens = _tokens(query)
    if not tokens:
        return []

    hits: list[dict] = []
    for p, meta, body in _scan():
        name = meta["name"].lower()
        desc = (meta.get("description", "") or "").lower()
        body_l = body.lower()
        score = 0.0

        # name 命中: 整个 query 作为 name 子串 (最强信号)
        q = query.lower()
        if q in name:
            score += _W_NAME * 2.0
        for t in tokens:
            if t in name:
                score += _W_NAME
            if t in desc:
                score += _W_DESC
            if t in body_l:
                score += _W_BODY
        if score <= 0:
            continue

        # 正文分数封顶, 防止长文档刷分
        score = min(score, _W_NAME * 2.0 + _W_DESC * len(tokens) + _BODY_CAP)
        try:
            updated = p.stat().st_mtime
        except OSError:
            updated = 0.0
        hits.append({
            "name": meta["name"],
            "description": meta.get("description", ""),
            "path": str(p),
            "score": round(score, 3),
            "updated": updated,
        })

    hits.sort(key=lambda h: -h["score"])
    return hits[:max(1, int(top_k))]
