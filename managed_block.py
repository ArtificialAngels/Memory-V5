"""受管块注入工具 —— marker 围栏 + 幂等替换 (借鉴 memU managed block).

用途: 向 SOUL.md / AGENTS.md / 任何用户文件注入一段"归我们管"的内容
(如检索指令), 同时保证:
  - 围栏标记 (``<!-- ikaros:begin -->`` ... ``<!-- ikaros:end -->``) 内是
    我们的, 重跑即升级 (幂等替换, 不叠加);
  - 围栏外是用户的, 永不动;
  - patch/strip 是纯函数, 无 I/O, 可单测; install/remove 才碰文件系统,
    写入前自动备份 .bak.

对应 memU 调研结论: "managed block 注入 (marker 围栏 + 重跑升级 + 用户内容
不动)" —— 落地为独立工具, 未来向 SOUL.md/AGENTS.md 注入检索指令时使用.
"""

from __future__ import annotations

import difflib
import re
import shutil
from pathlib import Path

BEGIN_MARKER = "<!-- ikaros:begin -->"
END_MARKER = "<!-- ikaros:end -->"

_BLOCK_RE = re.compile(
    rf"^{re.escape(BEGIN_MARKER)}\n.*?^{re.escape(END_MARKER)}\n?",
    re.DOTALL | re.MULTILINE,
)


def block(body: str) -> str:
    """组装完整受管块 (含围栏)."""
    return f"{BEGIN_MARKER}\n{body.rstrip()}\n{END_MARKER}\n"


def patch(current: str, body: str) -> str:
    """把受管块装入 ``current``: 已有则替换, 没有则追加.

    纯函数 (不碰文件系统), 幂等: 二次调用替换第一次的块而不是叠加.
    ``current`` 中围栏外的内容原样保留.
    """
    new_block = block(body)
    if _BLOCK_RE.search(current):
        return _BLOCK_RE.sub(lambda _: new_block, current, count=1)
    if current and not current.endswith("\n"):
        current += "\n"
    separator = "\n" if current else ""
    return f"{current}{separator}{new_block}"


def strip(current: str) -> str:
    """移除受管块, 其余内容原样保留. 纯函数, patch 的逆操作."""
    if not _BLOCK_RE.search(current):
        return current
    stripped = _BLOCK_RE.sub("", current, count=1)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    if not stripped.strip():
        return ""
    return stripped.rstrip("\n") + "\n"


def install(path: str | Path, body: str, *, backup: bool = True, dry_run: bool = False) -> tuple[bool, str]:
    """把受管块写入 ``path``. 返回 ``(changed, diff)``.

    写入前把原文件备份为 ``<path>.bak`` (目标文件属于用户, 可能含
    与我们无关的内容; 备份是安全网).
    """
    path = Path(path).expanduser()
    current = path.read_text(encoding="utf-8") if path.is_file() else ""
    updated = patch(current, body)
    diff = "".join(
        difflib.unified_diff(
            current.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
    )
    if updated == current or dry_run:
        return False, diff

    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and current:
        shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
    path.write_text(updated, encoding="utf-8")
    return True, diff


def remove(path: str | Path, *, backup: bool = True, dry_run: bool = False) -> tuple[bool, str]:
    """从 ``path`` 移除受管块 (install 的逆操作). 无块/无文件 = 干净 no-op."""
    path = Path(path).expanduser()
    if not path.is_file():
        return False, ""
    current = path.read_text(encoding="utf-8")
    updated = strip(current)
    diff = "".join(
        difflib.unified_diff(
            current.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
    )
    if updated == current or dry_run:
        return False, diff

    if backup and current:
        shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
    path.write_text(updated, encoding="utf-8")
    return True, diff
