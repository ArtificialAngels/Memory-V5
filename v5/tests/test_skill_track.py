"""skill_store / skill_tool 测试 (V5.5 skill track, 借鉴 memU 磁盘即接口).

验证:
  S1. write_skill 创建 + front-matter 渲染 + 原子写 (tmp 不残留)
  S2. write_skill 更新 (幂等, 内容替换)
  S3. name 校验 (kebab-case), description/content 空值拒绝
  S4. list_skills / get_skill / remove_skill (幂等 no-op)
  S5. search_skills 加权: name > description > body, 正文封顶, 返回无全文
  S6. 损坏文件 (无 front-matter) 静默跳过
  S7. 工具层 smoke: v5_skill_write / search / get / remove 走 answer 格式
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from v5 import skill_store
from v5.tools.skill_tool import (
    v5_skill_write, v5_skill_search, v5_skill_get, v5_skill_remove,
)


@pytest.fixture()
def skills(tmp_path, monkeypatch):
    """把技能目录指到临时目录, 隔离真实数据."""
    monkeypatch.setattr(skill_store, "SKILLS_DIR", tmp_path)
    return tmp_path


def _write(skills, name="deploy-windows", desc="Windows 部署流程", body="步骤 1: ...\n步骤 2: ..."):
    return skill_store.write_skill(name=name, description=desc, content=body)


# ── S1/S2: 写入与渲染 ──

def test_write_creates_file_with_frontmatter(skills):
    r = _write(skills)
    assert r["ok"] and r["created"] is True
    path = skills / "deploy-windows.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\nname: deploy-windows\ndescription: Windows 部署流程\n---\n")
    assert "步骤 1" in text
    # 原子写不残留 tmp
    assert not list(skills.glob("*.tmp"))


def test_write_updates_existing(skills):
    _write(skills)
    r = _write(skills, body="更新后的正文")
    assert r["created"] is False
    assert "更新后的正文" in skill_store.get_skill("deploy-windows")["content"]


# ── S3: 校验 ──

def test_name_validation(skills):
    with pytest.raises(ValueError):
        skill_store.write_skill("Bad Name", "d", "c")
    with pytest.raises(ValueError):
        skill_store.write_skill("中文名", "d", "c")
    with pytest.raises(ValueError):
        skill_store.write_skill("", "d", "c")


def test_empty_desc_or_content_rejected(skills):
    with pytest.raises(ValueError):
        skill_store.write_skill("ok-name", "", "content")
    with pytest.raises(ValueError):
        skill_store.write_skill("ok-name", "desc", "  ")


# ── S4: 列表 / 读取 / 删除 ──

def test_list_get_remove(skills):
    _write(skills)
    _write(skills, name="git-workflow", desc="Git 协作", body="rebase 优先")

    listed = skill_store.list_skills()
    assert {s["name"] for s in listed} == {"deploy-windows", "git-workflow"}
    assert all("content" not in s for s in listed)  # 列表不含全文

    got = skill_store.get_skill("git-workflow")
    assert got["description"] == "Git 协作" and "rebase" in got["content"]

    assert skill_store.remove_skill("git-workflow") is True
    assert skill_store.remove_skill("git-workflow") is False  # 幂等
    assert skill_store.get_skill("git-workflow") is None
    assert skill_store.get_skill("No Such Skill") is None  # 非法名不炸


# ── S5: 渐进形状检索 ──

def test_search_weighting(skills):
    _write(skills, name="deploy-windows", desc="Windows 部署流程", body="用 winget 安装")
    _write(skills, name="git-workflow", desc="Git 协作流程", body="用 rebase 整理提交")
    _write(skills, name="hermes-route", desc="Hermes 路由决策", body="聊天走 Hermes")

    # name 命中 > description 命中 > 正文命中
    hits = skill_store.search_skills("deploy")
    assert hits[0]["name"] == "deploy-windows"
    assert all("content" not in h for h in hits)  # 窄命中不含全文

    hits2 = skill_store.search_skills("Hermes")
    assert hits2[0]["name"] == "hermes-route"
    assert hits2[0]["score"] > 0

    # 中英混排必须能命中 (SOUL注入 → soul + 注入 拆开匹配)
    _write(skills, name="soul-guide", desc="SOUL 注入指南", body="往 SOUL.md 里写指令")
    hits3 = skill_store.search_skills("SOUL注入")
    assert hits3[0]["name"] == "soul-guide"

    # 空/无意义 query
    assert skill_store.search_skills("") == []
    assert skill_store.search_skills("x") == []  # 单字符 token 丢弃


def test_search_body_capped(skills):
    long_body = "word " * 200
    _write(skills, name="long-doc", desc="desc", body=long_body)
    _write(skills, name="short-doc", desc="desc word", body="brief")
    # body 命中分数封顶, 不会因为正文长而刷分到 name 命中之上
    hits = skill_store.search_skills("word")
    assert hits[0]["score"] <= skill_store._W_NAME * 2.0 + skill_store._W_DESC + skill_store._BODY_CAP + 1e-6


# ── S6: 损坏文件跳过 ──

def test_corrupt_file_skipped(skills):
    _write(skills)
    (skills / "not-a-skill.md").write_text("没有 front-matter 的普通文件", encoding="utf-8")
    listed = skill_store.list_skills()
    assert [s["name"] for s in listed] == ["deploy-windows"]


# ── S7: 工具层 smoke ──

def test_tool_smoke(skills):
    out = v5_skill_write("test-skill", "测试技能", "内容")
    assert "已创建" in out and '"ok": true' in out

    out = v5_skill_search("test")
    assert '"name": "test-skill"' in out and '"content"' not in out

    out = v5_skill_get("test-skill")
    assert '"content": "内容"' in out

    out = v5_skill_remove("test-skill")
    assert '"ok": true' in out
    # 再次删除 = 干净 no-op, 不抛错
    out2 = v5_skill_remove("test-skill")
    assert "not_found" in out2
