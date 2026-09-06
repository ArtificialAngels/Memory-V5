# 详细说明见 docs/v5-mcp-consolidation.md
"""memory_v5.loop — 标准记忆循环引擎 (Standard Memory Loop).

把「每轮必做的记忆仪式」从散落的 N 次 MCP 调用 / 插件 hook 收敛成一个声明式
阶段引擎: **一个 phase 一次调用跑完该阶段所有到期 step**。

## 为什么要它

2026-08-30 之前, 一轮对话的记忆动作分散在四处:

    dsh 插件 agent/pre-step        -> v5_recall / v5_context_refresh
    dsh 插件 agent/turn-stopping   -> v5_memory_store + (手工拼) 状态推进
    dsh 插件 ctx.interval (6h)     -> v5_call.py tick -> reflect scheduler
    agent 手动调用                 -> v5_vitality_tick / v5_relationship_tick /
                                      v5_anti_repeat_record / v5_reflect_run_op

后四个是**纯机器态维护动作**——模型不该决定要不要做、什么时候做, 却占着 4 个
MCP 工具槽位, 且漏调就是静默欠账 (实测: 手工 tick 从未被调过, long_term 长期为 0)。
Loop 引擎把它们内化: 按 phase 自动跑, 带冷却与状态落盘, 从 MCP 工具面移除。

## 三阶段

    pre          轮次开始: 身份锚定 + 记忆召回 + 项目经验召回
    post         轮次结束: 情感/活力/关系推进 + 反重复语料记录
    maintenance  周期维护: 反思管线 (默认 6h, 与 retention/promote 对齐)

## 设计原则

- **声明式**: 加一个 step = 加一行 ``register_step``, 不改调度代码。这与
  ``reflect/scheduler.py`` 的 ``ReflectOp`` 注册表模式一致——V3 那套「加第 5 个
  操作要复制 4 行 if 模板」的老毛病不再重犯。
- **绝不阻断**: step 失败收集进 ``errors`` 返回, 不抛异常。契约同 ``v5_call.py``:
  记忆是增强, 不是阻断会话的理由。
- **幂等 + 冷却**: 每个 step 自带 ``interval_sec`` (0 = 每轮都跑), 状态落盘到
  ``data/v5/loop_state.json``, 重复调用 / 进程重启后行为一致。
- **零 LLM 成本**: 默认 step 全是纯算法或状态推进, 一轮的开销是毫秒级。

## 用法

    from memory_v5.loop import run_phase, status
    run_phase("post", response="...")          # 跑完 post 阶段所有到期 step
    status()                                   # 每个 step 的 last_run / due / next
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("ikaros.memory.v5.loop")

# ─── 阶段定义 ────────────────────────────────────────────────────────
PHASE_PRE = "pre"
PHASE_POST = "post"
PHASE_MAINTENANCE = "maintenance"
PHASES = (PHASE_PRE, PHASE_POST, PHASE_MAINTENANCE)

# ─── 默认间隔 ────────────────────────────────────────────────────────
# maintenance 的 reflect 间隔与 ikaros-memory 插件原 6h tick 对齐
# (retention/promote 同为 6h), 行为不变。
DEFAULT_REFLECT_INTERVAL = 21600      # 6h

# ─── 状态持久化 ──────────────────────────────────────────────────────
# 锚 memory_v5 包根 (loop.py 直接位于 memory_v5/), 与 store.py 的
# MEM_ROOT/data/v5 同目录 (v5.db 旁边)。
# 教训 (2026-08-24): reflect/scheduler.py 曾用 Path(__file__).parent 锚到
# reflect/ 子目录 -> 状态写进 core/memory_v5/reflect/data/v5/ 孤儿路径。
# 嵌套子包里 __file__.parent 层级会变, 这里不能想当然。
_MEM_ROOT = Path(__file__).resolve().parent            # core/memory_v5/
_V5_DATA_DIR = _MEM_ROOT / "data" / "v5"
_STATE_FILE = _V5_DATA_DIR / "loop_state.json"


@dataclass
class LoopContext:
    """一次 phase 运行的输入。所有字段都有默认值, 调用方可只给需要的。"""
    phase: str = ""
    query: str = ""            # pre: 本轮用户消息 (召回用)
    response: str = ""         # post: 本轮助手回复 (反重复语料用)
    session_id: str = "default"
    character: str = ""
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LoopStep:
    """单个循环步骤的注册信息 (对比 reflect/scheduler.py 的 ReflectOp)。

    fn 签名: ``(ctx: LoopContext) -> Any``; 返回值进 results[name]。
    异常由引擎捕获进 errors[name], **不上抛**。
    """
    name: str
    fn: Callable[[LoopContext], Any]
    phase: str
    interval_sec: int = 0      # 0 = 每轮都跑
    enabled: bool = True

    @property
    def state_key(self) -> str:
        return f"{self.phase}.{self.name}"


# ─── 状态读写 ────────────────────────────────────────────────────────

def load_state(path: Path | None = None) -> dict[str, float]:
    """加载各 step 的 last_run 时间戳。损坏时显式 log 并返空 (不静默吞)。"""
    import json
    p = path or _STATE_FILE
    if p.exists():
        try:
            raw = json.loads(p.read_text("utf-8"))
            return {str(k): float(v) for k, v in raw.items()}
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            logger.warning("loop_state.json 损坏, 重新初始化: %s", e)
    return {}


def save_state(state: dict[str, float], path: Path | None = None) -> None:
    """原子写 + 滚动 .bak (机器态文件, 不设漂移拒写)。"""
    from memory_v5.file_store import atomic_write_json
    p = path or _STATE_FILE
    atomic_write_json(p, state, make_backup=True, validator=None)


# ─── 默认 step 实现 ──────────────────────────────────────────────────
# 全部惰性 import: loop.py 被 tools/facade.py 引用, 而 tools/* 又被
# tools/__init__ 收集, 模块级 import 会成环。

def _step_identity(ctx: LoopContext) -> dict:
    """pre: 身份 + 情感 + 关系快照 (原 v5_context_refresh)。"""
    from memory_v5.self_model import SelfModel
    from memory_v5.affect import AffectState
    from memory_v5.relationship import Relationship

    sm = SelfModel.load()
    affect = AffectState.load().decay()
    rel = Relationship.load()
    ident = sm.data.get("identity", {})
    beliefs = sm.data.get("beliefs", {})
    return {
        "name": ident.get("name", "伊卡洛斯"),
        "nature": ident.get("nature", "人造天使"),
        "creator": ident.get("creator", "哥哥"),
        "beliefs": {k: str(v)[:80] for k, v in beliefs.items()},
        "mood": affect.to_prompt() if hasattr(affect, "to_prompt") else "",
        "pleasure": round(getattr(affect, "pleasure", 0), 2),
        "arousal": round(getattr(affect, "arousal", 0), 2),
        "dominance": round(getattr(affect, "dominance", 0), 2),
        "relationship_depth": round(rel.depth, 2),
        "relationship_warmth": round(rel.warmth, 2),
        "relationship_stage": rel.stage() if hasattr(rel, "stage") else "",
        "curiosity": sm.get_curiosity(),
    }


def _step_recall(ctx: LoopContext) -> dict:
    """pre: 预算感知记忆召回 (原 v5_recall)。空 query 直接跳过, 不空转。"""
    if not (ctx.query or "").strip():
        return {"skipped": "empty_query"}
    from memory_v5.tools.recall_tool import v5_recall
    import json as _json
    raw = v5_recall(
        ctx.query,
        session_id=ctx.session_id or "default",
        include_dsh_only=bool(ctx.extra.get("include_dsh_only", True)),
    )
    # v5_recall 经 @safe_tool 包装, 恒返 "自然语言\nJSON" 两段
    body = raw.split("\n", 1)[1] if "\n" in raw else raw
    try:
        parsed = _json.loads(body)
    except (ValueError, TypeError):
        return {"raw": raw}
    return parsed if isinstance(parsed, dict) else {"data": parsed}


def _step_project(ctx: LoopContext) -> dict:
    """pre: 项目记忆召回 (原 v5_project_retrieve)。

    ⚠️ 为什么不能把整句 query 直接透传（2026-08-30 实测）：

        v5_project_retrieve 带 tags 走的是 memory_api.search 的**结构化
        SQLite 路径**（project_tool docstring: "结构化 tag 精确匹配, 无
        ChromaDB 依赖"），此时 query 不是语义查询，而是被拼成
        `content LIKE '%<整句话>%' OR tags LIKE '%<整句话>%'` 的**子串过滤**。
        把 "MCP 工具合并 Loop 循环" 这种整句自然语言当子串 → 恒 0 命中 →
        项目轨 29 条 decision/pitfall/convention 一条都浮不出来, 而返回的
        count=0 又被插件当"无内容"跳过注入 —— 静默失效, 无任何报错。

    正确语义：项目轨是**常驻背景**（决策/坑/约定，本来就该每轮可见），
    不是按 query 精确命中的检索结果。所以：
        1) 先用 query 试着收窄（命中则说明本轮确实在聊这个项目话题）
        2) 收窄后为空 → 退回项目全览（按 weight 降序取最重要的 top_k）
    这样既保留话题相关性，又保证项目轨永不静默清空。
    """
    from memory_v5.tools.project_tool import v5_project_retrieve
    import json as _json

    project = ctx.extra.get("project", "ikaros")
    top_k = int(ctx.extra.get("project_top_k", 5))
    include_dsh_only = bool(ctx.extra.get("include_dsh_only", True))
    query = (ctx.query or "").strip()

    def _retrieve(q):
        raw = v5_project_retrieve(
            project=project,
            query=q or None,
            top_k=top_k,
            include_dsh_only=include_dsh_only,
        )
        try:
            parsed = _json.loads(raw)
        except (ValueError, TypeError):
            return {"raw": raw}
        return parsed if isinstance(parsed, dict) else {"data": parsed}

    narrowed = _retrieve(query) if query else None
    if narrowed and narrowed.get("count"):
        narrowed["strategy"] = "query_narrowed"
        return narrowed
    # 收窄为空 / 无 query → 项目全览（按 weight 降序）
    overview = _retrieve(None)
    if isinstance(overview, dict):
        overview["strategy"] = "project_overview"
        if narrowed and not narrowed.get("count"):
            overview["narrowed_empty"] = True
    return overview


def _step_vitality(ctx: LoopContext) -> dict:
    """post: 推进精力模型一格 (原 v5_vitality_tick(conversation=True))。

    历史: 本函数曾写成「调两次 tick」绕过一个 bug —— 旧版
    `vitality.tick(conversation=True)` 会抑制**整段经过时间**的空闲恢复,
    导致每轮只减不增、精力单调抽干到 0。那个 bug 已于 2026-08-30 在
    `vitality.py` 里根治: 恢复改按「真空闲分钟数」计算
    (新增 `conversation_minutes` 参数), 与 `conversation` 标志脱钩。

    现在语义干净了, 一次调用即可:
        (a) 经过时间里的空闲部分 -> 恢复; 基础衰减 -> 消耗   (总是算)
        (b) 本轮对话的一次性消耗 (0.04) + 计数              (conversation=True)
    """
    from memory_v5.vitality import Vitality
    v = Vitality.load()
    v.tick(conversation=True)
    v.save()
    return {
        "vitality": round(v.vitality, 4),
        "label": v.label(),
        "conversation": True,
    }


def _step_relationship(ctx: LoopContext) -> dict:
    """post: 记录一次交互 (原 v5_relationship_tick)。"""
    from memory_v5.relationship import Relationship
    intensity = float(ctx.extra.get("intensity", 0.3))
    r = Relationship.load().record_interaction(intensity)
    r.save()
    return {
        "depth": round(r.depth, 4),
        "warmth": round(r.warmth, 4),
        "stage": r.stage(),
        "closeness": round(r.closeness(), 4),
    }


def _step_anti_repeat(ctx: LoopContext) -> dict:
    """post: 把本轮回复记进反重复语料 (原 v5_anti_repeat_record)。

    P3-3 (2026-09-05): record 后同步检测重复风险, 若命中则返回 penalty_hint,
    供 dsh 插件注入下一轮 system prompt 提醒换话题。
    """
    if not (ctx.response or "").strip():
        return {"skipped": "empty_response"}
    from memory_v5 import anti_repeat
    character = ctx.character or "ikaros"
    n = anti_repeat.record_response(character, ctx.response)
    result = {"recorded": n}
    # 检测本轮回复是否与历史高度重复
    try:
        hint = anti_repeat.get_penalty_hint(character, ctx.response)
        if hint:
            result["penalty_hint"] = hint
    except Exception as e:
        # 检测失败不影响 record, 静默降级
        import logging
        logging.getLogger("ikaros.v5.loop").debug(
            "anti_repeat penalty check failed: %s", e)
    return result


def _step_reflect(ctx: LoopContext) -> dict:
    """maintenance: 反思管线 run_all (各 op 按自身间隔到期才跑)。"""
    from memory_v5.reflect.registry import make_default_scheduler
    from memory_v5.reflect.scheduler import load_state as _load, save_state as _save

    sched = make_default_scheduler(_load())
    results = sched.run_all(force=False, continue_on_error=True)
    return {"results": results}


# ─── 引擎 ────────────────────────────────────────────────────────────

class LoopEngine:
    """标准记忆循环引擎: 注册表模式, 按 phase 批量跑到期 step。

    对比 reflect/scheduler.py 的 ReflectScheduler:
      - 同样注册表 + 冷却 + 状态落盘
      - 不同点: 按 phase 分组 (一轮可能跑多个 phase), 且 step 接收 LoopContext
        (调度器 op 是无参 fn), 失败收集而非上抛。
    """

    def __init__(self, steps: list[LoopStep] | None = None,
                 state: dict[str, float] | None = None):
        self._steps: list[LoopStep] = list(steps) if steps is not None else _default_steps()
        self._state: dict[str, float] = dict(state) if state is not None else load_state()

    # ── 注册 ──
    def register(self, step: LoopStep) -> None:
        """注册 step. 同名覆盖 (设计选择, 同 ReflectScheduler.register)。"""
        if step.phase not in PHASES:
            raise ValueError(f"unknown phase: {step.phase!r} (must be one of {PHASES})")
        self._steps = [s for s in self._steps if s.state_key != step.state_key]
        self._steps.append(step)

    def steps(self, phase: str | None = None) -> list[LoopStep]:
        if phase is None:
            return list(self._steps)
        return [s for s in self._steps if s.phase == phase]

    @property
    def state(self) -> dict[str, float]:
        return dict(self._state)

    # ── 执行 ──
    def due_steps(self, phase: str, *, now: float | None = None,
                  force: bool = False) -> list[LoopStep]:
        """列出该 phase 当前该跑的 step (不改 state)。"""
        if now is None:
            now = time.time()
        out = []
        for s in self.steps(phase):
            if not s.enabled:
                continue
            if force or s.interval_sec <= 0:
                out.append(s)
                continue
            if (now - self._state.get(s.state_key, 0.0)) >= s.interval_sec:
                out.append(s)
        return out

    def run(self, phase: str, ctx: LoopContext | None = None, *,
            force: bool = False, now: float | None = None,
            state_path: Path | None = None) -> dict:
        """跑一个 phase 的所有到期 step。

        返回:
            {"ok": True, "phase": ..., "ran": [...], "skipped": {...},
             "errors": {...}, "results": {...}, "elapsed_ms": int}

        单个 step 失败 -> 进 errors[name], 其余 step 照跑 (continue-on-error)。
        这是桥接层契约: 记忆是增强, 不是阻断会话的理由。
        """
        if phase not in PHASES:
            return {"ok": False, "error": f"unknown phase: {phase!r}",
                    "phases": list(PHASES)}
        if now is None:
            now = time.time()
        if ctx is None:
            ctx = LoopContext()
        ctx.phase = phase

        t0 = time.time()
        ran: list[str] = []
        skipped: dict[str, str] = {}
        errors: dict[str, str] = {}
        results: dict[str, Any] = {}

        all_steps = self.steps(phase)
        due = self.due_steps(phase, now=now, force=force)
        due_keys = {s.state_key for s in due}
        for s in all_steps:
            if s.state_key not in due_keys:
                skipped[s.name] = (
                    "disabled" if not s.enabled
                    else f"cooldown({s.interval_sec}s)"
                )

        for s in due:
            try:
                out = s.fn(ctx)
                results[s.name] = out
                ran.append(s.name)
                self._state[s.state_key] = time.time()
            except Exception as e:  # noqa: BLE001 — 契约: 不上抛
                logger.error("loop step %s 失败: %s", s.state_key, e)
                errors[s.name] = f"{type(e).__name__}: {e}"

        try:
            save_state(self._state, state_path)
        except Exception as e:  # noqa: BLE001 — 状态落盘失败不该废掉本次结果
            logger.error("loop_state 落盘失败: %s", e)
            errors["_state"] = f"{type(e).__name__}: {e}"

        return {
            "ok": not errors,
            "phase": phase,
            "ran": ran,
            "skipped": skipped,
            "errors": errors,
            "results": results,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    # ── 观测 ──
    def status(self, *, now: float | None = None) -> dict:
        """每个 step 的 due / last_run / next_run_in_sec。给哥哥看的一屏状态。"""
        if now is None:
            now = time.time()
        out: dict[str, dict] = {}
        for phase in PHASES:
            out[phase] = {}
            for s in self.steps(phase):
                last = self._state.get(s.state_key, 0.0)
                next_at = last + s.interval_sec if s.interval_sec > 0 else now
                out[phase][s.name] = {
                    "enabled": s.enabled,
                    "due": (
                        s.enabled and (
                            s.interval_sec <= 0
                            or (now - last) >= s.interval_sec
                        )
                    ),
                    "last_run": last if last > 0 else None,
                    "last_run_human": (
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last))
                        if last > 0 else "never"
                    ),
                    "next_run_in_sec": (
                        0 if s.interval_sec <= 0
                        else max(0, int(next_at - now))
                    ),
                    "interval_sec": s.interval_sec,
                }
        return out


def _default_steps() -> list[LoopStep]:
    """默认 step 表 —— 加一个 step 就是这里加一行。"""
    return [
        # ── pre: 轮次开始 ──
        LoopStep("identity", _step_identity, PHASE_PRE),
        LoopStep("recall", _step_recall, PHASE_PRE),
        LoopStep("project", _step_project, PHASE_PRE),
        # ── post: 轮次结束 (原 4 个机器态 MCP 工具内化到这里) ──
        LoopStep("vitality", _step_vitality, PHASE_POST),
        LoopStep("relationship", _step_relationship, PHASE_POST),
        LoopStep("anti_repeat", _step_anti_repeat, PHASE_POST),
        # ── maintenance: 周期维护 ──
        LoopStep("reflect", _step_reflect, PHASE_MAINTENANCE,
                 interval_sec=DEFAULT_REFLECT_INTERVAL),
    ]


def make_default_engine(state: dict[str, float] | None = None) -> LoopEngine:
    return LoopEngine(steps=_default_steps(), state=state)


# ─── 模块级便捷入口 (v5_call.py / 插件走这两个) ───────────────────────

def run_phase(phase: str, **ctx_kwargs) -> dict:
    """跑一个 phase。ctx_kwargs 见 LoopContext 字段。

    force=True 忽略冷却全跑 (调试 / 手动补账用)。
    """
    force = bool(ctx_kwargs.pop("force", False))
    ctx = LoopContext(**{k: v for k, v in ctx_kwargs.items()
                         if k in LoopContext.__dataclass_fields__})
    extra = ctx_kwargs.get("extra")
    if isinstance(extra, dict):
        ctx.extra = extra
    return make_default_engine().run(phase, ctx, force=force)


def status() -> dict:
    """全 phase 的 step 状态快照。"""
    return make_default_engine().status()
