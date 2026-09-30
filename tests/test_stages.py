"""各 stage 的单元测试（不依赖 demo 数据）。"""

from __future__ import annotations

import pytest

from datasetfactory.lineage import DatasetDAG
from datasetfactory.schema import Sample
from datasetfactory.stages import (
    contamination,
    dedup,
    filter as filter_stage,
    fingerprint,
    ingest,
    normalize,
    quality,
    split,
)

NORM_CFG = {"unicode_form": "NFKC", "strip_html": True, "collapse_whitespace": True, "drop_empty": True}


# ----------------------------------------------------------------------
# ingest
# ----------------------------------------------------------------------
def _ingest_cfg(root, sources):
    return {"_root": str(root), "sources": sources}


def test_ingest_jsonl_reads_rows_and_registers_source(tmp_path):
    (tmp_path / "d.jsonl").write_text(
        '{"text": "第一条内容"}\n{"text": "第二条内容"}\n', encoding="utf-8"
    )
    cfg = _ingest_cfg(tmp_path, [{"source_id": "d", "kind": "jsonl", "uri": "d.jsonl"}])
    dag = DatasetDAG()

    samples, sources = ingest.ingest(cfg, dag)

    assert len(samples) == 2
    assert sources[0].source_id == "d"
    assert all(s.lineage == ["src:d"] for s in samples)
    assert dag.nodes["src:d"].stats["samples"] == 2


def test_ingest_remote_kind_reaches_loader_instead_of_path_check(tmp_path):
    """hf/web 的 uri 是标识符不是路径，不该被本地存在性检查提前拦下。"""
    cfg = _ingest_cfg(
        tmp_path, [{"source_id": "hf", "kind": "hf", "uri": "datasets/foo"}]
    )
    with pytest.raises(NotImplementedError, match="hf loader"):
        ingest.ingest(cfg, DatasetDAG())


def test_ingest_missing_local_file_names_the_path(tmp_path):
    cfg = _ingest_cfg(
        tmp_path, [{"source_id": "w", "kind": "jsonl", "uri": "nope.jsonl"}]
    )
    with pytest.raises(FileNotFoundError, match="找不到文件"):
        ingest.ingest(cfg, DatasetDAG())


def test_ingest_unknown_kind_lists_available_kinds(tmp_path):
    cfg = _ingest_cfg(tmp_path, [{"source_id": "x", "kind": "s3", "uri": "s3://b/k"}])
    with pytest.raises(ValueError, match="可用"):
        ingest.ingest(cfg, DatasetDAG())


# ----------------------------------------------------------------------
# normalize
# ----------------------------------------------------------------------
def test_normalize_strips_html_and_unescapes():
    out = normalize.normalize_text("<p>Hello &amp; <b>world</b></p>", NORM_CFG)
    assert out == "Hello & world"


def test_normalize_collapses_whitespace():
    out = normalize.normalize_text("a\t\t b\n\n\n\nc", NORM_CFG)
    assert out == "a b\n\nc"


def test_normalize_removes_zero_width():
    out = normalize.normalize_text("a\u200bb\ufeffc", NORM_CFG)
    assert out == "abc"


def test_normalize_assigns_unique_ids_to_duplicates():
    """内容相同的两条记录：content_digest 相同，但 sample_id 必须唯一。"""
    dag = DatasetDAG()
    dag.add_node(__import__("datasetfactory.lineage", fromlist=["LineageNode"]).LineageNode("src:a", "source", "A"))
    raw = [
        Sample(sample_id="x", text="同样的内容", source_id="a", lineage=["src:a"]),
        Sample(sample_id="y", text="同样的内容", source_id="a", lineage=["src:a"]),
    ]
    out = normalize.normalize(raw, NORM_CFG, dag)

    assert len(out) == 2
    assert out[0].sample_id != out[1].sample_id
    assert out[1].sample_id.endswith("-2")
    assert out[0].meta["content_digest"] == out[1].meta["content_digest"]


def test_normalize_drops_empty():
    dag = DatasetDAG()
    from datasetfactory.lineage import LineageNode

    dag.add_node(LineageNode("src:a", "source", "A"))
    raw = [
        Sample(sample_id="x", text="   ", source_id="a", lineage=["src:a"]),
        Sample(sample_id="y", text="有内容", source_id="a", lineage=["src:a"]),
    ]
    out = normalize.normalize(raw, NORM_CFG, dag)
    assert len(out) == 1
    assert dag.nodes["xf:normalize"].stats["empty_dropped"] == 1


# ----------------------------------------------------------------------
# fingerprint
# ----------------------------------------------------------------------
def test_simhash_identical_text_has_zero_distance():
    a = fingerprint.simhash(fingerprint.shingles("这是一段用于测试的文本内容"), 64)
    b = fingerprint.simhash(fingerprint.shingles("这是一段用于测试的文本内容"), 64)
    assert fingerprint.hamming(a, b) == 0


def test_simhash_similar_text_has_small_distance():
    base = "混合专家模型通过路由器为每个 token 选择 Top-K 个专家参与计算，核心难点在于负载均衡。"
    tweaked = "混合专家模型通过路由器为每个 token 选择 Top-K 个专家参与计算，核心难点是负载均衡问题。"
    far = "梯度下降沿着损失函数的负梯度方向更新参数，学习率决定每次更新的步长。"

    d_near = fingerprint.hamming(
        fingerprint.simhash(fingerprint.shingles(base)),
        fingerprint.simhash(fingerprint.shingles(tweaked)),
    )
    d_far = fingerprint.hamming(
        fingerprint.simhash(fingerprint.shingles(base)),
        fingerprint.simhash(fingerprint.shingles(far)),
    )
    assert d_near < d_far


def test_shingles_ignores_whitespace():
    assert fingerprint.shingles("a b c d e", 5) == ["abcde"]
    assert fingerprint.shingles("", 5) == []
    assert fingerprint.shingles("ab", 5) == ["ab"]


# ----------------------------------------------------------------------
# dedup
# ----------------------------------------------------------------------
def _mk(sid: str, text: str) -> Sample:
    s = Sample(sample_id=sid, text=text, source_id="a")
    s.fingerprint = fingerprint.simhash(fingerprint.shingles(text))
    return s


# 标定用长文本（187 个 shingle）。SimHash 的汉明距离强烈依赖文本长度：
# 同一处改动，短文本（~40 shingle）会产生 2 倍于长文本的距离。
# 实测（见 dedup.py 文档）：
#     删 1 个字 → 2    改 1 个词 → 6    改 2 个词 → 9    改 3 个词 → 13
#     无关文本（同长度） → 34
_BASE = (
    "混合专家模型用一组并行的专家网络替代稠密前馈层，并通过路由器为每个 token 选择 "
    "Top-K 个专家参与计算。由于只有少数专家被激活，模型总参数量可以远大于单次前向"
    "实际使用的参数量，从而在几乎不增加计算量的前提下扩大模型容量。MoE 的核心难点"
    "在于负载均衡：如果路由器总是偏好少数专家，就会出现专家塌缩，多数专家得不到"
    "有效训练。常用的解法是引入辅助损失、容量因子以及对专家并行的通信优化。"
)

_UNRELATED = (
    "数据库索引结构决定了查询的代价模型。B+ 树把数据按页组织，叶子节点用链表串联以"
    "支持范围扫描；哈希索引只适合等值查询，无法服务区间条件。查询优化器会基于统计"
    "信息估算选择率，再在嵌套循环、哈希连接与排序归并之间选型。"
)


def test_simhash_distance_calibration():
    """固化标定数字：长文本改 1 词应远小于默认阈值，无关文本应远大于阈值。"""
    base_fp = fingerprint.simhash(fingerprint.shingles(_BASE))
    one_word_fp = fingerprint.simhash(fingerprint.shingles(_BASE.replace("替代", "取代")))
    unrelated_fp = fingerprint.simhash(fingerprint.shingles(_UNRELATED))

    d_near = fingerprint.hamming(base_fp, one_word_fp)
    d_far = fingerprint.hamming(base_fp, unrelated_fp)

    assert d_near <= 8, f"改 1 个词的距离为 {d_near}，默认阈值 8 应能覆盖"
    assert d_far > 25, f"无关文本的距离为 {d_far}，不该靠近任何合理阈值"
    assert d_near < d_far


def test_dedup_removes_exact_duplicate():
    dag = DatasetDAG()
    samples = [_mk("a:1", "完全一样的内容"), _mk("a:2", "完全一样的内容")]
    kept, rejected = dedup.deduplicate(samples, {"exact": True, "near": {"enabled": False}}, dag)

    assert len(kept) == 1
    assert len(rejected) == 1
    assert rejected[0].rule == "exact"
    assert rejected[0].detail["duplicate_of"] == "a:1"


def test_dedup_near_duplicate_within_threshold():
    """阈值内的一词改动应被判为 near 重复。"""
    dag = DatasetDAG()
    kept, rejected = dedup.deduplicate(
        [_mk("a:1", _BASE), _mk("a:2", _BASE.replace("替代", "取代"))],
        {"exact": True, "near": {"enabled": True, "hamming_threshold": 12}},
        dag,
    )
    assert len(kept) == 1
    assert len(rejected) == 1
    assert rejected[0].rule == "near"
    assert rejected[0].detail["hamming"] <= 12
    assert rejected[0].detail["duplicate_of"] == "a:1"


def test_dedup_keeps_unrelated_text_at_same_threshold():
    """同一阈值下无关文本不能被误合并 —— 防止放宽阈值时静默丢数据。"""
    dag = DatasetDAG()
    kept, rejected = dedup.deduplicate(
        [_mk("a:1", _BASE), _mk("a:2", _UNRELATED)],
        {"exact": True, "near": {"enabled": True, "hamming_threshold": 12}},
        dag,
    )
    assert len(kept) == 2
    assert rejected == []


def test_dedup_keeps_distinct_samples():
    dag = DatasetDAG()
    samples = [_mk("a:1", "第一个完全不同的主题内容，讲的是数据库索引结构。"),
               _mk("a:2", "第二个完全不同的主题内容，讲的是网络协议分层模型。")]
    kept, rejected = dedup.deduplicate(samples, {"exact": True, "near": {"enabled": True, "hamming_threshold": 6}}, dag)
    assert len(kept) == 2
    assert rejected == []


# ----------------------------------------------------------------------
# quality
# ----------------------------------------------------------------------
QCFG = {
    "min_chars": 40,
    "length_target": 400,
    "weights": {"length": 0.30, "content_ratio": 0.30, "line_uniqueness": 0.25, "symbol_penalty": 0.15},
}


def test_quality_hard_fails_short_text():
    detail = quality.score_text("???", QCFG)
    assert detail["score"] == 0.0
    assert detail["hard_fail"] == "min_chars"


def test_quality_scores_normal_text_higher_than_symbol_soup():
    good = "这是一段长度足够、内容比例正常的中文文本，用于验证质量打分函数的排序能力。" * 2
    soup = "§§§◆◆◆" * 10

    assert quality.score_text(good, QCFG)["score"] > quality.score_text(soup, QCFG)["score"]


def test_quality_penalises_repeated_lines():
    unique = "\n".join(f"第{i}行内容各不相同，用于测试行唯一性指标。" for i in range(5))
    repeated = "\n".join(["重复的一行内容，用于测试行唯一性指标。"] * 5)

    d_unique = quality.score_text(unique, QCFG)
    d_repeated = quality.score_text(repeated, QCFG)
    assert d_unique["line_uniqueness"] > d_repeated["line_uniqueness"]


def test_quality_sets_expected_metrics():
    dag = DatasetDAG()
    from datasetfactory.lineage import LineageNode

    dag.add_node(LineageNode("src:a", "source", "A"))
    s = Sample(sample_id="a:1", text="一段足够长的正常中文文本，用来验证指标写入是否正确。" * 2,
               source_id="a", lineage=["src:a"])
    out = quality.score_quality([s], QCFG, dag)

    for name in ("quality.score", "quality.chars", "quality.content_ratio",
                 "quality.line_uniqueness", "quality.symbol_penalty"):
        assert out[0].has_metric(name), name


# ----------------------------------------------------------------------
# contamination
# ----------------------------------------------------------------------
def test_ngrams_basic():
    assert contamination.ngrams("", 3) == set()
    assert contamination.ngrams("ab", 3) == {"ab"}
    assert contamination.ngrams("abcd", 2) == {"ab", "bc", "cd"}


def test_contamination_flags_verbatim_copy():
    pool = contamination.ngrams("这是一段会被评测集收录的标准答案文本内容。", 5)
    grams = contamination.ngrams("这是一段会被评测集收录的标准答案文本内容。", 5)
    assert len(grams & pool) / len(grams) == pytest.approx(1.0)


def test_contamination_no_overlap_for_unrelated_text():
    pool = contamination.ngrams("完全无关的另一段文本，讲的是网络协议。", 8)
    grams = contamination.ngrams("这段内容讨论的是数据库索引与查询优化。", 8)
    assert len(grams & pool) == 0


# ----------------------------------------------------------------------
# filter
# ----------------------------------------------------------------------
def test_filter_evaluate_missing_metric():
    s = Sample(sample_id="a:1", text="x", source_id="a")
    passed, value, why = filter_stage.evaluate(s, {"metric": "quality.score", "op": ">=", "value": 0.5})
    assert passed is False
    assert value is None
    assert why == "metric_missing"


def test_filter_records_evidence_for_kept_sample():
    dag = DatasetDAG()
    from datasetfactory.lineage import LineageNode

    dag.add_node(LineageNode("src:a", "source", "A"))
    s = Sample(sample_id="a:1", text="x", source_id="a", lineage=["src:a"])
    s.set_metric("quality.score", 0.9)

    kept, rejected = filter_stage.apply_filter(
        [s], {"filter_id": "f", "on_fail": "drop",
              "rules": [{"metric": "quality.score", "op": ">=", "value": 0.5}]}, dag
    )
    assert len(kept) == 1 and rejected == []
    evidence = kept[0].meta["filter_evidence"]["f"]
    assert evidence[0]["passed"] is True
    assert evidence[0]["actual"] == pytest.approx(0.9)


def test_filter_rejects_and_records_reason():
    dag = DatasetDAG()
    from datasetfactory.lineage import LineageNode

    dag.add_node(LineageNode("src:a", "source", "A"))
    s = Sample(sample_id="a:1", text="x", source_id="a", lineage=["src:a"])
    s.set_metric("quality.score", 0.1)

    kept, rejected = filter_stage.apply_filter(
        [s], {"filter_id": "f", "on_fail": "drop",
              "rules": [{"metric": "quality.score", "op": ">=", "value": 0.5}]}, dag
    )
    assert kept == []
    assert rejected[0].detail["actual"] == pytest.approx(0.1)
    assert "quality.score >= 0.5" in rejected[0].rule


def test_filter_rejects_unknown_op():
    dag = DatasetDAG()
    s = Sample(sample_id="a:1", text="x", source_id="a")
    s.set_metric("quality.score", 0.9)
    with pytest.raises(ValueError, match="unsupported filter op"):
        filter_stage.apply_filter(
            [s], {"filter_id": "f", "rules": [{"metric": "quality.score", "op": "~=", "value": 0.5}]}, dag
        )


# ----------------------------------------------------------------------
# split
# ----------------------------------------------------------------------
def test_split_is_deterministic():
    ratios = {"train": 0.7, "valid": 0.15, "test": 0.15}
    a = split.assign_split("wiki:abc123", ratios, 2)
    b = split.assign_split("wiki:abc123", ratios, 2)
    assert a == b
    assert a in ratios


def test_split_covers_all_samples():
    ratios = {"train": 0.7, "valid": 0.15, "test": 0.15}
    names = {split.assign_split(f"id:{i}", ratios, 2) for i in range(500)}
    assert names == {"train", "valid", "test"}


def test_bucket_is_in_unit_interval():
    for i in range(200):
        assert 0.0 <= split.bucket(f"id:{i}", 42) < 1.0
