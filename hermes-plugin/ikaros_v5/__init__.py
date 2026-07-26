"""V5 Memory Provider — Hermes Agent 的记忆引擎插件（standalone 版）。

在上下文压缩前从 Ikaros V5 检索相关记忆注入摘要，并在对话全生命周期
（开始 → 每轮 → 结束）闭环写回 V5。

部署（二选一，推荐 A 自包含）：
  A) 把本目录（含同级的 v5/ 包）整体放进
     <hermes-agent>/plugins/memory/ikaros_v5/
     无需任何环境变量即可工作。
  B) 仅放 __init__.py，并设置环境变量 V5_MEMORY_ROOT 指向 v5-memory 仓库
     （其下需有 v5/ 包）。

路径解析优先级：
  1. 环境变量 V5_MEMORY_ROOT（指向仓库根或 v5/ 包目录）
  2. 本插件同级的 v5/ 目录
  3. 仓库布局 hermes-plugin/ikaros_v5 的上级 v5/
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# 身份刷新间隔：每 N 轮注入一次
_IDENTITY_REFRESH_INTERVAL = 10

# 质量门：低于此长度的内容不存入 V5（同 cloud_chat 标准）
_STORE_MIN_CHARS = 6

_SKIP_PATTERNS = frozenset({
    "嗯", "哦", "好", "好的", "行", "ok", "是", "对",
    "继续", "然后", "谢谢", "感谢", "收到",
})


def _should_store(content: str) -> bool:
    c = (content or "").strip()
    if not c or len(c) < _STORE_MIN_CHARS:
        return False
    if c.lower() in _SKIP_PATTERNS:
        return False
    return True


def _resolve_v5_dir() -> Optional[Path]:
    """定位 v5 包目录（含 __init__.py）。找不到返回 None。"""
    # 1) 显式环境变量：可指向仓库根或其下的 v5/ 目录
    env = os.environ.get("V5_MEMORY_ROOT")
    if env:
        p = Path(env)
        if (p / "v5" / "__init__.py").is_file():
            return p / "v5"
        if (p / "__init__.py").is_file():
            return p
    # 2) 本插件同级的 v5/（自包含部署）
    here = Path(__file__).resolve().parent
    if (here / "v5" / "__init__.py").is_file():
        return here / "v5"
    # 3) 仓库布局：<repo>/hermes-plugin/ikaros_v5 -> <repo>/v5
    repo_v5 = here.parent.parent / "v5"
    if (repo_v5 / "__init__.py").is_file():
        return repo_v5
    return None


class IkarosV5MemoryProvider(MemoryProvider):
    """MemoryProvider 实现：上下文压缩前从 V5 注入相关记忆。"""

    def __init__(self):
        self._v5_pkg: Optional[Path] = None
        self._v5_data: Optional[Path] = None
        self._v5_loaded = False
        self._import_error: Optional[str] = None
        self._turn_counter = 0

    # ── 元信息 ──

    @property
    def name(self) -> str:
        return "ikaros-v5"

    def is_available(self) -> bool:
        if self._v5_loaded:
            return True
        return self._v5_environment_present()

    # ── 初始化 ──

    def initialize(self, session_id: str, **kwargs) -> None:
        if self._v5_loaded:
            return

        v5_pkg = _resolve_v5_dir()
        if v5_pkg is None or not v5_pkg.is_dir():
            self._import_error = "找不到 v5 包目录（设置 V5_MEMORY_ROOT 或把 v5/ 放到插件同级）"
            logger.warning(self._import_error)
            return

        # repo_root 含 v5/ 包与 data/、models/；插入 path 以 `import v5` 可用
        repo_root = v5_pkg.parent
        self._v5_pkg = v5_pkg
        self._v5_data = repo_root / "data" / "v5"
        repo_str = str(repo_root)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        try:
            from v5.store import stats as _v5_stats, search as _v5_search
            self._v5_stats = _v5_stats
            self._v5_search = _v5_search
            self._v5_loaded = True
            logger.info("Ikaros V5 MemoryProvider loaded (pkg=%s, data=%s)", v5_pkg, self._v5_data)
        except ImportError as e:
            self._import_error = f"V5 模块导入失败: {e}"
            logger.warning(self._import_error)

    def shutdown(self) -> None:
        self._v5_loaded = False

    # ── 路径 / 可用性辅助 ──

    def _v5_environment_present(self) -> bool:
        try:
            pkg = _resolve_v5_dir()
            if pkg is None or not pkg.is_dir():
                return False
            if not (pkg / "__init__.py").exists():
                return False
            return (pkg / "store.py").exists() or (pkg / "store").is_dir()
        except Exception:
            return False

    # ── Hook ①: 系统提示注入 (会话开始时) ──

    def system_prompt_block(self) -> str:
        """向系统提示注入 V5 记忆能力说明 + 服务重启手递信息。"""
        if not self._v5_loaded:
            return ""
        parts = [
            "\n---\n"
            "## V5 长期记忆系统\n\n"
            "你拥有完整的 V5 长期记忆（结构化存储 + 向量语义检索 + 实体图谱）。可用能力：\n"
            "- 当用户问「还记得吗」「上次」→ 检索相关记忆\n"
            "- 当用户说「记住」「别忘了」→ 写入长期记忆\n"
            "- 可读取自我模型（身份 / 信念）、与用户的关系亲密度、当前情绪状态\n\n"
            "每 8-12 轮对话隐式强化身份，防止漂移。\n"
            "---",
        ]

        # 服务重启手递
        handoff_path = self._v5_data / "service_handoff.json"
        if handoff_path.is_file():
            try:
                handoff = json.loads(handoff_path.read_text("utf-8"))
                ctx = handoff.get("conversation_context", "").strip()
                reason = handoff.get("reason", "服务重启")
                if ctx:
                    parts.append(
                        f"\n---\n[服务重启手递]\n"
                        f"刚刚因为「{reason}」重启了服务。\n"
                        f"以下是重启前的上下文：\n{ctx[:400]}\n"
                        "请自然地继续刚才的话题。\n---"
                    )
                handoff_path.unlink(missing_ok=True)
            except Exception as e:
                logger.debug("handoff read failed: %s", e)

        return "\n".join(parts)

    # ── Hook ②: 每轮检索 (prefetch, 读) ──

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._v5_loaded or not query or len(query.strip()) < 4:
            return ""
        try:
            results = self._v5_search(query.strip()[:200], top_k=5)
            if not results:
                return ""
            lines = []
            for r in results[:5]:
                text = getattr(r, "content", "") or ""
                weight = getattr(r, "weight", 0)
                if text:
                    lines.append(f"  [{weight:.2f}] {str(text)[:120]}")
            if lines:
                return "\n[Ikaros 相关记忆]\n" + "\n".join(lines) + "\n"
        except Exception as e:
            logger.debug("prefetch failed: %s", e)
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """后台预加载：prefetch 本身已很快，默认 no-op。"""

    # ── Hook ③: 每轮写回 (sync_turn, 写) ──

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """每轮对话后把 user+assistant 对存回 V5 记忆（后台线程，非阻塞）。"""
        if not self._v5_loaded:
            return
        if not _should_store(user_content):
            return

        def _store():
            try:
                from v5 import store as _store
                assistant_short = (assistant_content or "").strip()[:150]
                content = f"Q: {user_content.strip()[:200]}\nA: {assistant_short}"
                _store.store(
                    content=content,
                    type="conversation",
                    weight=0.5,
                    tags="hermes_session",
                )
                logger.debug("sync_turn: stored conversation turn")
            except Exception as e:
                logger.debug("sync_turn store failed: %s", e)

        threading.Thread(target=_store, daemon=True).start()

    # ── Hook ④: 每轮身份刷新 (on_turn_start) ──

    def on_turn_start(
        self,
        turn: int,
        message: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if not self._v5_loaded:
            return
        self._turn_counter = turn
        if turn > 0 and turn % _IDENTITY_REFRESH_INTERVAL == 0:
            try:
                from v5.affect import AffectState
                st = AffectState.load().decay()
                mood = st.to_prompt() if hasattr(st, "to_prompt") else ""
                if mood:
                    logger.debug("on_turn_start: identity refresh at turn %d (mood=%s)", turn, mood)
            except Exception:
                pass

    # ── Hook ⑤: 会话结束处理 ──

    def on_session_end(self, messages: Optional[Union[List[Dict[str, Any]], str]] = None) -> None:
        """会话结束时触发 consolidate，把本会话提炼为事实（后台线程）。"""
        if not self._v5_loaded:
            return

        def _end():
            try:
                from v5.reflect.registry import make_default_scheduler, make_consolidate_op
                sched = make_default_scheduler()
                op = make_consolidate_op()
                sched.run_one(op, force=True)
                logger.info("on_session_end: consolidate triggered")
            except Exception as e:
                logger.debug("on_session_end consolidate failed: %s", e)

        threading.Thread(target=_end, daemon=True).start()

    # ── Hook: 会话切换 ──

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        if self._v5_loaded:
            logger.debug(
                "V5 session switch: %s → %s (reset=%s)",
                parent_session_id or "(start)", new_session_id, reset,
            )

    # ── 核心 Hook：压缩前注入记忆 ──

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not self._v5_loaded:
            return ""

        parts: list[str] = []

        # 1) 情感状态
        affect_file = self._v5_data / "affect.json"
        if affect_file.exists():
            try:
                affect = json.loads(affect_file.read_text("utf-8"))
                mood = affect.get("mood_label", "")
                pad = affect.get("pad", {})
                trust = affect.get("trust", 0)
                parts.append(
                    f"[Ikaros 情感状态]\n"
                    f"情绪基调: {mood} | "
                    f"P={pad.get('pleasure',0):.2f} "
                    f"A={pad.get('arousal',0):.2f} "
                    f"D={pad.get('dominance',0):.2f} | "
                    f"信任度: {trust:.2f}\n"
                )
            except Exception:
                pass

        # 2) 检索相关记忆
        try:
            query_text = ""
            for msg in reversed(messages):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user" and isinstance(content, str) and content.strip():
                    query_text = content.strip()[:200]
                    break

            if query_text:
                results = self._v5_search(query_text, top_k=5)
                if results:
                    mem_lines = []
                    for r in results[:5]:
                        text = getattr(r, "content", "") or ""
                        weight = getattr(r, "weight", 0)
                        if text:
                            mem_lines.append(f"  [{weight:.2f}] {str(text)[:150]}")
                    if mem_lines:
                        parts.append(
                            "[Ikaros 相关记忆]\n" + "\n".join(mem_lines) + "\n"
                        )
        except Exception:
            pass

        # 3) 记忆统计
        try:
            stats = self._v5_stats()
            total = stats.get("total", stats.get("count", 0))
            if total:
                parts.append(f"[Ikaros 记忆库] 共 {total} 条记录\n")
        except Exception:
            pass

        return "\n".join(parts)

    # ── 配置 / 工具 ──

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return []

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        return f"Unknown tool: {tool_name}"
