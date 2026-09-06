"""managed_block 受管块注入测试 (借鉴 memU managed block 设计).

验证:
  M1. patch: 无块时追加; 有块时替换 (幂等, 不叠加)
  M2. strip: patch 的逆操作, round-trip 字节级还原
  M3. 围栏外用户内容原样保留
  M4. install/remove 文件系统: backup 生成, 缺文件 no-op
  M5. 多行 body 含换行/标记字符安全
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from memory_v5 import managed_block as mb


BODY = "## memU 检索指令\n\nRun `memu-hermes retrieve` before answering.\n"


# ── M1: patch 幂等 ──

def test_patch_appends_when_absent():
    current = "# 用户文件\n原有内容\n"
    out = mb.patch(current, BODY)
    assert out.startswith("# 用户文件\n原有内容\n")
    assert mb.BEGIN_MARKER in out and mb.END_MARKER in out
    assert out.count(mb.BEGIN_MARKER) == 1


def test_patch_replaces_idempotent():
    current = mb.patch("# 用户\n", "第一版")
    twice = mb.patch(current, "第二版")
    assert twice.count(mb.BEGIN_MARKER) == 1
    assert "第一版" not in twice
    assert "第二版" in twice


def test_patch_empty_current():
    out = mb.patch("", BODY)
    assert out.startswith(mb.BEGIN_MARKER)
    assert out.count(mb.BEGIN_MARKER) == 1


# ── M2: strip 逆操作 ──

def test_strip_roundtrip():
    original = "# 用户文件\n\n原有内容\n\n更多内容\n"
    patched = mb.patch(original, BODY)
    restored = mb.strip(patched)
    assert restored == original


def test_strip_no_block_noop():
    assert mb.strip("无块内容") == "无块内容"
    assert mb.strip("") == ""


# ── M3: 用户内容保留 ──

def test_user_content_preserved():
    current = "A\nB\n"
    patched = mb.patch(current, BODY)
    assert "A\nB\n" in patched
    assert mb.strip(patched) == "A\nB\n"


# ── M4: 文件系统 ──

def test_install_creates_with_backup(tmp_path):
    target = tmp_path / "SOUL.md"
    target.write_text("原有", encoding="utf-8")

    changed, diff = mb.install(target, BODY)
    assert changed and diff
    assert (tmp_path / "SOUL.md.bak").read_text(encoding="utf-8") == "原有"
    text = target.read_text(encoding="utf-8")
    assert mb.BEGIN_MARKER in text

    # 重跑 = 升级不叠加
    changed2, _ = mb.install(target, "升级版")
    assert text.count(mb.BEGIN_MARKER) == 1
    assert "升级版" in target.read_text(encoding="utf-8")


def test_install_dry_run_no_write(tmp_path):
    target = tmp_path / "x.md"
    changed, diff = mb.install(target, BODY, dry_run=True)
    assert changed is False and diff
    assert not target.exists()


def test_remove_noop_when_missing(tmp_path):
    target = tmp_path / "none.md"
    changed, _ = mb.remove(target)
    assert changed is False


def test_remove_restores_original(tmp_path):
    target = tmp_path / "AGENTS.md"
    original = "项目说明\n"
    target.write_text(original, encoding="utf-8")
    mb.install(target, BODY)
    mb.remove(target)
    assert target.read_text(encoding="utf-8") == original


# ── M5: 特殊字符安全 ──

def test_body_with_braces_and_markers():
    body = "含 {braces} 与 <!-- 注释 --> 的正文\n第二行"
    current = mb.patch("# 头", body)
    assert "{braces}" in current
    assert mb.strip(current) == "# 头\n"
