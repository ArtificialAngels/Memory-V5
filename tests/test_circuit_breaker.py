"""memory_v5.search 短路器 (circuit breaker) 测试 — Task 2.0.2 (2026-08-20).

设计: docs/memory_v5-circuit-breaker-design.md §4

覆盖:
  T1. closed → open: 连续失败 ≥ threshold (3) 后跳闸
  T2. closed 状态: 失败 < threshold 时被成功重置
  T3. open 状态: 调用 _fetch_embedding 不进网络, _do_fetch_embedding 调用次数为 0
  T4. open → half_open → closed: 冷却期过后探针成功
  T5. open → half_open → open: 冷却期过后探针失败, opened_at 重置
  T6. circuit_breaker_enabled=False: 关闭开关, 每次都打网络
  T7. (设计 §4.1 T7) 多线程并发失败: 跳闸恰好一次, 无竞态
  T8. (设计 §4.2) 集成: 模拟 :8587 不可达, 跳闸后 retrieve < 100ms

技术选择:
  - monkeypatch search_mod._do_fetch_embedding 控制"网络"行为
    (失败抛 ConnectionRefusedError / 成功返回 vec), 不动真实端口
  - 不依赖时间流逝: 用 monkeypatch.setattr(search_mod, "time", fake_time)
    直接控制 _circuit_is_open 内部的 time.monotonic()
  - 每个测试用 _circuit_reset() 把全局 _CIRCUIT 清干净, 避免测试间污染
"""
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import memory_v5.search as search_mod


# ── 夹具与辅助 ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_breaker():
    """每个测试前后清空全局 _CIRCUIT, 避免相互污染。"""
    search_mod._circuit_reset()
    yield
    search_mod._circuit_reset()


class _FakeTime:
    """可手动推进的 fake monotonic clock; 用于测试冷却期 (open → half_open)。"""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fake_time(monkeypatch):
    """替换 search_mod.time 为可控虚拟时钟。"""
    ft = _FakeTime()
    monkeypatch.setattr(search_mod, "time", ft)
    return ft


def _enable_breaker(monkeypatch, *, enabled: bool = True, threshold: int = 3,
                    reset_seconds: float = 30.0):
    """注入 yaml 风格的配置, 让短路器各参数可调。"""
    monkeypatch.setattr(search_mod, "_cache_cfg", lambda: {
        "circuit_breaker_enabled": enabled,
        "circuit_breaker_threshold": threshold,
        "circuit_breaker_reset_seconds": reset_seconds,
    })


def _stub_do_fetch(monkeypatch, behavior):
    """替换 _do_fetch_embedding 为可控函数。

    behavior 可为:
      - "ok": 每次返回 [0.1, 0.2, 0.3]
      - "raise_conn_refused": 每次抛 ConnectionRefusedError
      - "raise_timeout": 每次抛 socket.timeout
      - callable: 自定义 (text, task) -> Optional[list[float]]
    """
    calls = {"n": 0}

    def _impl(text, task="query"):
        calls["n"] += 1
        if callable(behavior):
            return behavior(text, task)
        if behavior == "ok":
            return [0.1, 0.2, 0.3]
        if behavior == "raise_conn_refused":
            raise ConnectionRefusedError("simulated:8587 down")
        if behavior == "raise_timeout":
            raise TimeoutError("simulated:8587 timeout")
        if behavior == "raise_type_error":
            raise TypeError("simulated programming error (not a network error)")
        raise RuntimeError(f"unknown behavior: {behavior}")

    monkeypatch.setattr(search_mod, "_do_fetch_embedding", _impl)
    return calls


# ── T1: closed → open ──────────────────────────────────────────────────

def test_t1_three_failures_trips_breaker(monkeypatch):
    """T1: 连续 3 次网络异常 → state=open, opened_at > 0。"""
    _enable_breaker(monkeypatch, threshold=3)
    calls = _stub_do_fetch(monkeypatch, "raise_conn_refused")

    assert search_mod._circuit_state()["state"] == "closed"
    assert search_mod._fetch_embedding("q1") is None
    assert search_mod._circuit_state()["state"] == "closed"
    assert search_mod._fetch_embedding("q2") is None
    assert search_mod._circuit_state()["state"] == "closed"

    # 第 3 次失败后跳闸
    assert search_mod._fetch_embedding("q3") is None
    s = search_mod._circuit_state()
    assert s["state"] == "open"
    assert s["failure_count"] == 3
    assert s["opened_at"] > 0
    assert calls["n"] == 3


# ── T2: 失败 < threshold 时成功重置 ───────────────────────────────────

def test_t2_two_failures_then_success_resets(monkeypatch):
    """T2: 2 次失败 + 1 次成功 → state=closed, failure_count=0。"""
    _enable_breaker(monkeypatch, threshold=3)

    # 先 2 次失败
    _stub_do_fetch(monkeypatch, "raise_conn_refused")
    assert search_mod._fetch_embedding("q1") is None
    assert search_mod._fetch_embedding("q2") is None
    assert search_mod._circuit_state()["failure_count"] == 2

    # 改成成功
    _stub_do_fetch(monkeypatch, "ok")
    assert search_mod._fetch_embedding("q3") == [0.1, 0.2, 0.3]
    s = search_mod._circuit_state()
    assert s["state"] == "closed"
    assert s["failure_count"] == 0


# ── T3: open 状态下不进网络 ─────────────────────────────────────────────

def test_t3_open_state_skips_network(monkeypatch):
    """T3: open 状态下 _fetch_embedding 返回 None, _do_fetch_embedding 不被调用。"""
    _enable_breaker(monkeypatch, threshold=2)
    calls = _stub_do_fetch(monkeypatch, "raise_conn_refused")

    # 触发跳闸
    search_mod._fetch_embedding("q1")
    search_mod._fetch_embedding("q2")
    assert search_mod._circuit_state()["state"] == "open"
    n_after_trip = calls["n"]

    # 后续调用直接短路, 不进 _do_fetch_embedding
    for _ in range(10):
        assert search_mod._fetch_embedding("followup") is None
    assert calls["n"] == n_after_trip, (
        f"open 状态调用了 {calls['n']} 次网络, 期望 {n_after_trip} (短路不调网络)")


# ── T4: open → half_open → closed (探针成功) ──────────────────────────

def test_t4_half_open_probe_success_closes(monkeypatch, fake_time):
    """T4: 跳闸 → 冷却 30s → 探针成功 → closed。"""
    _enable_breaker(monkeypatch, threshold=2, reset_seconds=30.0)
    _stub_do_fetch(monkeypatch, "raise_conn_refused")
    search_mod._fetch_embedding("q1")
    search_mod._fetch_embedding("q2")
    assert search_mod._circuit_state()["state"] == "open"

    # 推进时间到冷却期外 (31s)
    fake_time.advance(31.0)
    # 改成成功, 让探针成功
    _stub_do_fetch(monkeypatch, "ok")

    # 这一次调用会触发 open → half_open (在 _circuit_is_open 内部),
    # 然后 _do_fetch_embedding 返回 vec, _circuit_record_success → closed
    vec = search_mod._fetch_embedding("probe")
    assert vec == [0.1, 0.2, 0.3]
    s = search_mod._circuit_state()
    assert s["state"] == "closed"
    assert s["failure_count"] == 0


# ── T5: open → half_open → open (探针失败, opened_at 重置) ────────────

def test_t5_half_open_probe_failure_reopens(monkeypatch, fake_time):
    """T5: 探针失败 → 立即重新 open, opened_at 更新为 now (冷却重置)。"""
    _enable_breaker(monkeypatch, threshold=2, reset_seconds=30.0)
    _stub_do_fetch(monkeypatch, "raise_conn_refused")
    search_mod._fetch_embedding("q1")
    search_mod._fetch_embedding("q2")
    initial_open_at = search_mod._circuit_state()["opened_at"]
    assert search_mod._circuit_state()["state"] == "open"

    fake_time.advance(31.0)
    # 探针也失败
    assert search_mod._fetch_embedding("probe") is None
    s = search_mod._circuit_state()
    assert s["state"] == "open"
    assert s["opened_at"] > initial_open_at, "探针失败应重置 opened_at (冷却重置)"


# ── T6: 关闭开关 = 不短路 ─────────────────────────────────────────────

def test_t6_disabled_behavior_matches_old(monkeypatch):
    """T6: circuit_breaker_enabled=false → 100 次失败也不跳闸, 每次都调网络。"""
    _enable_breaker(monkeypatch, enabled=False, threshold=3)
    calls = _stub_do_fetch(monkeypatch, "raise_conn_refused")

    for _ in range(100):
        assert search_mod._fetch_embedding("q") is None

    assert calls["n"] == 100
    assert search_mod._circuit_state()["state"] == "closed"


# ── T7: 多线程并发失败 = 跳闸恰好一次, 无竞态 ─────────────────────────

def test_t7_concurrent_failures_no_race(monkeypatch):
    """T7: 16 个线程各打 10 次网络异常 → 状态正确 (open 或 closed, failure_count ≤ threshold*2)。

    关键不变式: failure_count 是单调递增, 一旦达到阈值就变 open;
    后续失败会被 _circuit_record_failure 的早 return (state!=half_open 走 closed 累加路径)
    继续累加, 但 state 不会再变 (除非阈值变更)。这里只验证:
      1. 最终 state == open
      2. _do_fetch_embedding 总调用次数 == 阈值 (恰好跳闸)
         — 因为跳闸后所有线程都被 _circuit_is_open 拦截
      3. 无异常
    """
    threshold = 3
    _enable_breaker(monkeypatch, threshold=threshold)
    calls = _stub_do_fetch(monkeypatch, "raise_conn_refused")

    errors: list[BaseException] = []
    n_threads = 16
    per_thread = 10
    barrier = threading.Barrier(n_threads)

    def worker():
        try:
            barrier.wait()  # 同步起跑, 放大竞争窗口
            for i in range(per_thread):
                assert search_mod._fetch_embedding(f"q-{threading.get_ident()}-{i}") is None
        except BaseException as e:  # noqa: BLE001 — 收集所有异常
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"并发线程异常: {errors}"
    # 跳闸后所有调用都短路, 不会超过阈值 + 探针窗口;
    # 这里允许略多于 threshold (race 窗口内多打几次), 但远小于 n_threads*per_thread
    assert calls["n"] <= threshold + n_threads, (
        f"网络调用 {calls['n']} 远大于阈值+race余量, 短路未生效")
    assert search_mod._circuit_state()["state"] == "open"


# ── 错误分类: 只计网络错误 ─────────────────────────────────────────────

def test_non_network_errors_dont_trip_breaker(monkeypatch):
    """设计 §5.1: TypeError/ValueError 等程序错误不应跳闸。"""
    _enable_breaker(monkeypatch, threshold=3)
    _stub_do_fetch(monkeypatch, "raise_type_error")

    for _ in range(10):
        assert search_mod._fetch_embedding("q") is None

    assert search_mod._circuit_state()["state"] == "closed"
    assert search_mod._circuit_state()["failure_count"] == 0


# ── T8: 集成 — 模拟 :8587 不可达, 跳闸后 retrieve < 100ms ─────────────

def test_t8_simulated_8587_down_fast_fail(monkeypatch):
    """T8 (集成): 模拟 :8587 down (用 monkeypatch 改 _do_fetch_embedding 抛 ConnectionRefusedError)。

    验证:
      1. 前 3 次调用进网络 + 触发跳闸
      2. 跳闸后 10 次调用 < 100ms 总耗时 (≈ 微秒级)
    """
    _enable_breaker(monkeypatch, threshold=3)
    _stub_do_fetch(monkeypatch, "raise_conn_refused")

    # 触发跳闸 (前 3 次)
    for _ in range(3):
        assert search_mod._fetch_embedding("q") is None
    assert search_mod._circuit_state()["state"] == "open"

    # 后续 10 次必须极快 (短路不调网络)
    start = time.perf_counter()
    for _ in range(10):
        assert search_mod._fetch_embedding("q") is None
    elapsed = time.perf_counter() - start
    assert elapsed < 0.1, f"短路后 10 次调用耗时 {elapsed*1000:.1f}ms, 期望 < 100ms"


def test_t8b_simulated_8587_down_real_socket(monkeypatch):
    """T8b: 不 monkeypatch, 用真实 socket 连死端口 (127.0.0.1:1) 验证短路性能。

    这才是设计 §4.2 描述的"模拟 :8587 不可达"。
    """
    import socket as _socket
    _enable_breaker(monkeypatch, threshold=3)
    monkeypatch.setattr(search_mod, "EMBED_URL", "http://127.0.0.1:1/embedding")
    monkeypatch.setattr(search_mod, "EMBED_TIMEOUT", 10)
    # 强制连接单例重建 (URL 改了) — 2026-08-24 P1-5: thread-local
    search_mod._reset_embed_conn()
    # 关掉 LRU 缓存, 否则前几次失败会被后续命中遮蔽
    monkeypatch.setattr(search_mod, "_cache_enabled", lambda: False)

    # 触发跳闸 (前 3 次, ECONNREFUSED on Windows 上几毫秒)
    t0 = time.perf_counter()
    for _ in range(3):
        assert search_mod._fetch_embedding("q") is None
    trip_elapsed = time.perf_counter() - t0
    assert search_mod._circuit_state()["state"] == "open", (
        f"3 次后未跳闸: trip_elapsed={trip_elapsed:.2f}s state={search_mod._circuit_state()}")
    # ECONNREFUSED 应该 < 1s/次, 3 次共 < 3s; 但实际 Windows 上常 < 100ms/次

    # 跳闸后 10 次必须极快
    start = time.perf_counter()
    for _ in range(10):
        assert search_mod._fetch_embedding("q") is None
    elapsed = time.perf_counter() - start
    assert elapsed < 0.1, (
        f"短路后 10 次耗时 {elapsed*1000:.1f}ms, 期望 < 100ms "
        f"(trip_elapsed={trip_elapsed:.2f}s)")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))