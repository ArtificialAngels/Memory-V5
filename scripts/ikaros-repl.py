# 详细说明见 docs/scripts/bin/ikaros-repl.md
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# 让 import 找到 cloud_chat / cogno_5d
def _ikaros_root() -> Path:
    env = os.environ.get("IKAROS_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for d in (here, *here.parents):
        if (d / "core" / "memory_v5").is_dir():
            return d
    return here.parents[2] if len(here.parents) > 2 else here

_ROOT = _ikaros_root()
for p in (str(_ROOT / "bin"),
         str(_ROOT / "core/memory_v5")):
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
log = logging.getLogger("ikaros.repl")


def _load_cogno_5d():
    """加载 Ikaros-memory/cogno_5d 真物 (f40d838 修后)."""
    try:
        from cogno_5d import enrich, enrich_reply, reset_context
        return enrich, enrich_reply, reset_context
    except Exception as e:
        log.warning("cogno_5d not available (live chat without 5D): %s", e)
        # 失败静默: 返 dummy 包装 (不破坏 chat 主链路)
        def enrich(text, history=None):
            return f"[Cogno noop: {text[:30]}]"
        def enrich_reply(reply, *args, **kwargs):
            return {"reply": reply, "cogno": "[noop]"}
        def reset_context():
            return None
        return enrich, enrich_reply, reset_context


def _load_cloud_chat():
    """加载 bin/cloud_chat 真物 (cogno 5D + cloud_llm 接通)."""
    # cloud_chat.py 已于 2026-08-13 废弃删除，恒返回 None（回退本地 LLM）。
    return None


def _load_goal_contract():
    """加载 Ikaros-memory/goal_contract 真物 (借自 Hermes Agent /goal draft (2026-08-18 退役, 代码内联保留)).

    返回 (draft_fn, GoalContract) 或 (None, None) — 失败静默, 不破坏主链.
    """
    try:
        import goal_contract
        return goal_contract.draft_contract, goal_contract.GoalContract
    except Exception as e:
        log.warning("goal_contract not available: %s", e)
        return None, None


def _call_llm(prompt: str, enrich_prefix: str) -> str:
    """调用 LLM, 优先 cloud_chat, 回退 :8080 llama-server."""
    cloud_chat_fn = _load_cloud_chat()
    if cloud_chat_fn is not None:
        # cloud_chat 异步接口 or sync -- 试 sync first
        try:
            result = cloud_chat_fn(prompt, session_id="ikaros_repl_001")
            if hasattr(result, "__await__"):
                import asyncio
                result = asyncio.run(result)
            if isinstance(result, dict):
                return result.get("reply") or result.get("content") or str(result)
            return str(result)
        except Exception as e:
            log.warning("cloud_chat call failed: %s", e)
    # Fallback: 直接打 :8080 llama-server
    import json
    import urllib.request
    body = json.dumps({
        "model": "local-llm",
        "messages": [{"role": "system", "content": "You are Ikaros (人造天使). Reply briefly, 80-120 chars, Chinese."},
                     {"role": "user", "content": enrich_prefix + "\n\n" + prompt}],
        "max_tokens": 600,
        "temperature": 0.7,
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "ikaros-repl/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        d = json.loads(resp.read().decode("utf-8"))
    return d["choices"][0]["message"]["content"]


def main() -> int:
    enrich, enrich_reply, reset_context = _load_cogno_5d()
    reset_context()  # 清缓存

    print("=" * 72)
    print("伊卡洛斯 (人造天使, codename Alpha) -- CLI 直聊")
    print("=" * 72)
    print("输入文本后回车得到回答. Ctrl-C 退出. > 后面是你打的字.")
    print("斜杠命令: /contract <目标>  -- 把自然语言目标扩写成结构化合同 (5 字段)")
    print()

    while True:
        try:
            user_text = input("> 哥哥说: ").strip()
        except EOFError:
            print()
            print("(EOF, exit)")
            break
        except KeyboardInterrupt:
            print()
            print("(Ctrl-C, exit)")
            break

        if not user_text:
            continue
        if user_text in ("exit", "quit", ":q"):
            print("(exit)")
            break

        # 0) 斜杠命令: /contract <objective>
        #    把自然语言目标扩写成结构化合同 (outcome/verification/constraints/
        #    boundaries/stop_when). 不走 cogno + cloud_chat, 单 LLM 调用.
        if user_text.startswith("/contract"):
            draft_fn, _GoalContract = _load_goal_contract()
            if draft_fn is None:
                print("伊卡洛斯: (goal_contract 不可用, 无法扩写)")
                print()
                continue
            objective = user_text[len("/contract"):].strip()
            if not objective:
                print("伊卡洛斯: 用法: /contract <目标>")
                print()
                continue
            t0 = time.time()
            try:
                contract = draft_fn(objective)
            except Exception as e:
                print(f"伊卡洛斯: (draft 失败: {type(e).__name__}: {e})")
                print()
                continue
            dt_ms = int((time.time() - t0) * 1000)
            if contract is None:
                print(f"伊卡洛斯: (扩写失败或返回空合同, {dt_ms}ms)")
                print()
                continue
            print(f"伊卡洛斯 (合同 {dt_ms}ms):")
            print(contract.render_block())
            print()

        # 1) cogno 5 维 enrich system prompt (前 250 chars)
        cogno_prefix = enrich(user_text, history=None)
        if not cogno_prefix or not cogno_prefix.startswith("【认知5D】"):
            cogno_prefix = "【认知5D】 " + cogno_prefix

        # 2) 调 LLM
        t0 = time.time()
        try:
            reply = _call_llm(user_text, cogno_prefix)
        except Exception as e:
            print(f"伊卡洛斯: (LLM call failed: {type(e).__name__}: {e})")
            continue
        dt_ms = int((time.time() - t0) * 1000)

        # 3) cogno 5 维 enrich_reply (Phase 5 返 dict)
        try:
            tag = enrich_reply(reply, user_text)
            if isinstance(tag, dict):
                cogno_tag = tag.get("cogno", "[unknown]")
            else:
                cogno_tag = str(tag)
        except Exception as e:
            cogno_tag = f"[enrich_reply failed: {type(e).__name__}]"

        # 4) Print reply
        print(f"伊卡洛斯 ({dt_ms}ms, {cogno_tag}): {reply}")
        print()


if __name__ == "__main__":
    sys.exit(main())
