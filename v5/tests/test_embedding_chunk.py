"""search._fetch_embedding 分块嵌入测试 (V5.6, 2026-08-10).

背景: :8587 llama-server 物理批限制 ≈512 tokens, 超长输入返回 HTTP 500,
长记忆 (对话转写) 向量同步静默失败 → FTS-only, 召回受损.
修复: >350 字符分块嵌入 + mean pooling.

验证:
  E1. 短文本单次请求, 返回 mock 维度
  E2. 长文本 (2000+ 字符) 拆成多块, document 无前缀 (bge-m3), 结果平均池化
  E3. 任一块 HTTP 500 → 整体返回 None (fail-open)
  E4. 空文本 → None
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

import v5.search as search_mod


class _FakeResponse:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    def read(self):
        return self._payload


class _FakeConn:
    """记录请求, 按块返回不同向量."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[bytes] = []

    def request(self, method, path, body, headers=None):
        self.calls.append(body)

    def getresponse(self):
        return self._responses.pop(0)


class _FakeHTTPConnection:
    def __init__(self, host, port=80, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def request(self, method, path, body, headers=None):
        self._conn.request(method, path, body, headers=headers)

    def getresponse(self):
        return self._conn.getresponse()

    def close(self):
        pass


def _vec(dim, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return [float(x) for x in (v / np.linalg.norm(v))]


def _make_payload(vec):
    # :8587 实际返回形状: [{"index":0, "embedding":[[...]]}]
    return json.dumps([{"index": 0, "embedding": [vec]}]).encode("utf-8")


@pytest.fixture(autouse=True)
def _patch_http(monkeypatch):
    """替换 http.client.HTTPConnection (函数内 import 拿系统级模块), 记录请求体."""

    class Recorder:
        instances: list = []
        responses: list = []

        def __init__(self, host, port=80, timeout=None):
            self.host = host
            self.port = port
            self.timeout = timeout
            self.calls: list[bytes] = []
            Recorder.instances.append(self)

        def request(self, method, path, body, headers=None):
            self.calls.append(body)

        def getresponse(self):
            return _FakeResponse(200, _make_payload(_vec(64, len(Recorder.instances))))

        def close(self):
            pass

    monkeypatch.setattr("http.client.HTTPConnection", Recorder)
    yield Recorder


# ── E1: 短文本单次请求 ──

def test_e1_short_text_single_call(_patch_http):
    vec = search_mod._fetch_embedding("短文本", task="query")
    assert vec is not None and len(vec) == 64
    assert len(_patch_http.instances) == 1
    body = json.loads(_patch_http.instances[0].calls[0])
    assert body["content"].startswith("为这个句子生成表示以用于检索相关文章：短文本")


# ── E2: 长文本分块 + 平均池化 ──

def test_e2_long_text_chunked_and_pooled(_patch_http):
    long_text = "这是一段非常长的中文记忆内容。" * 100  # 1300+ 字符
    vec = search_mod._fetch_embedding(long_text, task="document")
    assert vec is not None and len(vec) == 64
    n_calls = len(_patch_http.instances)
    assert n_calls >= 3  # 1300 字符 / 350 → 至少 4 块 (保险断言 3+)
    # bge-m3: document 不加前缀 (query 才加检索指令)
    for inst in _patch_http.instances:
        body = json.loads(inst.calls[0])
        assert not body["content"].startswith("为这个句子生成表示")
    # 均值池化: 结果是各块向量的逐元素平均 (随机单位向量平均后范数 < 1, 属预期)
    assert len(vec) == 64
    assert 0.0 < float(np.linalg.norm(np.asarray(vec))) < 1.0


# ── E3: 任一块 500 → None ──

def test_e3_chunk_failure_fail_open(monkeypatch):
    class FailOnce:
        count = 0

        def __init__(self, host, port=80, timeout=None):
            pass

        def request(self, method, path, body, headers=None):
            pass

        def getresponse(self):
            FailOnce.count += 1
            if FailOnce.count == 2:  # 第二块失败
                return _FakeResponse(500, b'{"error":"too large"}')
            return _FakeResponse(200, _make_payload(_vec(64, 1)))

        def close(self):
            pass

    monkeypatch.setattr("http.client.HTTPConnection", FailOnce)
    long_text = "x" * 900
    assert search_mod._fetch_embedding(long_text, task="document") is None


# ── E4: 空文本 → None ──

def test_e4_empty_text_none(_patch_http):
    assert search_mod._fetch_embedding("", task="query") is None
    assert search_mod._fetch_embedding("   ", task="query") is None
    assert len(_patch_http.instances) == 0
