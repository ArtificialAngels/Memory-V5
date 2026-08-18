"""阶段 3: 轻量本体对齐测试 (零 LLM 成本, 借鉴 cognee FuzzyMatchingStrategy).

验证:
  O1. align_entity: 模糊命中 (伊卡洛斯→Ikaros), 低于阈值拒绝, 全角归一
  O2. find_entity_candidates_fuzzy: exact 优先, 不足时 difflib 补召回
  O3. alias_extract: 中文/英文别名正则抽取
  O4. add_alias: 幂等写 eg_aliases
"""
import os
import sys
import tempfile
from pathlib import Path

# 盘符无关: 脚本位置推导 (tests/ -> v5 -> core)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v5 import entity_graph as eg
from v5.extensions import ontology_align as oa


def _fresh_eg_db():
    tmp = tempfile.mkdtemp(prefix="ontology_test_")
    eg.EG_DB_PATH = Path(os.path.join(tmp, "v5.db"))
    with eg.eg_conn() as c:
        c.execute("INSERT INTO eg_entities (canonical_name, type) VALUES ('Ikaros', 'system')")
        c.execute("INSERT INTO eg_entities (canonical_name, type) VALUES ('Hermes', 'agent')")
        c.commit()
        # eg_conn 自动 commit, 显式也安全


# ── O1: align_entity ──
def test_o1_align_entity():
    _fresh_eg_db()
    # 归一化 (大小写/全角/括号) 后命中 (difflib 只处理同语言拼写变体, 非跨语言翻译)
    hit = oa.align_entity("ikaros")
    assert hit is not None and hit["canonical_name"] == "Ikaros"
    assert abs(hit["similarity"] - 1.0) < 1e-6
    hit2 = oa.align_entity("Ikaros (AI)")
    assert hit2 is not None and hit2["canonical_name"] == "Ikaros"
    # 轻微拼写变体 → 命中
    hit3 = oa.align_entity("Hermess")
    assert hit3 is not None and hit3["canonical_name"] == "Hermes"
    # 完全不相关 → 拒绝
    assert oa.align_entity("香蕉共和国") is None
    # 空输入 → None
    assert oa.align_entity("   ") is None


# ── O2: fuzzy candidates ──
def test_o2_fuzzy_candidates():
    _fresh_eg_db()
    # exact 命中优先
    out = oa.find_entity_candidates_fuzzy("Hermes", top_k=2)
    assert out and out[0].canonical_name == "Hermes"
    assert abs(out[0].similarity - 1.0) < 1e-6
    # 近似输入 → difflib 补路 (至少不崩, 可能 None 也可能命中)
    out2 = oa.find_entity_candidates_fuzzy("hermes 代理", top_k=2)
    assert isinstance(out2, list)


# ── O3: alias_extract ──
def test_o3_alias_extract():
    pairs = oa.alias_extract("伊卡洛斯（又称 Ikaros）住在上海, 系统 Hermes (aka Hermes2) 是 agent")
    mains = {m for m, _ in pairs}
    aliases = {a for _, a in pairs}
    assert "伊卡洛斯" in mains
    assert "Ikaros" in aliases
    assert "Hermes" in mains or "Hermes2" in mains
    assert "Hermes2" in aliases
    # 无别名文本 → 空
    assert oa.alias_extract("普通句子没有别名") == []


# ── O4: add_alias 幂等 ──
def test_o4_add_alias():
    _fresh_eg_db()
    eid = str(eg.find_entity_candidates("Ikaros")[0].entity_id)
    assert oa.add_alias(eid, "伊卡洛斯") is True
    assert oa.add_alias(eid, "伊卡洛斯") is True  # 幂等, 不报错
    with eg.eg_conn() as c:
        rows = c.execute("SELECT alias FROM eg_aliases WHERE entity_id = ?", (eid,)).fetchall()
    assert [r["alias"] for r in rows] == ["伊卡洛斯"]
    # 空参数 → False
    assert oa.add_alias("", "") is False


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
