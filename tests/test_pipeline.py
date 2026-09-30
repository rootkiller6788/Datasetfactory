"""端到端流水线测试：跑真实 demo 数据，验证结构不变量与血缘可追溯性。"""

from __future__ import annotations

import json

import pytest

from datasetfactory.pipeline import run_pipeline
from datasetfactory.report import build_report, explain_sample, write_artifacts


# ----------------------------------------------------------------------
# 漏斗结构不变量（不写死具体条数，避免改数据就挂）
# ----------------------------------------------------------------------
def test_funnel_is_monotonic(result):
    c = result.counters
    chain = [
        c["raw"],
        c["normalized"],
        c["fingerprinted"],
        c["deduped"],
        c["quality_scored"],
        c["contamination_checked"],
        c["filtered"],
        c["total_final"],
    ]
    assert all(a >= b for a, b in zip(chain, chain[1:])), chain


def test_every_rejection_is_accounted_for(result):
    """丢弃数 = 各阶段差额之和。"""
    c = result.counters
    assert c["raw"] - c["deduped"] == sum(
        1 for r in result.rejections if r.stage == "dedup"
    )
    assert c["deduped"] - c["filtered"] == sum(
        1 for r in result.rejections if r.stage == "filter"
    )
    assert len(result.rejections) == c["raw"] - c["total_final"]


def test_all_splits_sum_to_final_count(result):
    assert sum(len(v) for v in result.dataset.splits.values()) == result.counters["total_final"]


def test_sample_ids_are_unique_across_kept_and_rejected(result):
    kept = [s.sample_id for s in result.dataset.all_samples()]
    dropped = [r.sample_id for r in result.rejections]
    assert len(kept) == len(set(kept)), "保留样本内存在重复 id"
    assert not (set(kept) & set(dropped)), "同一个 id 同时出现在保留与丢弃两侧"


# ----------------------------------------------------------------------
# demo 数据应触发的具体行为
# ----------------------------------------------------------------------
def test_exact_duplicate_was_detected(result):
    exact = [r for r in result.rejections if r.stage == "dedup" and r.rule == "exact"]
    assert exact, "demo 数据里 wiki.jsonl 有一对完全相同的记录，应被 exact 判重"
    assert exact[0].detail["duplicate_of"]


def test_near_duplicate_was_detected(result):
    near = [r for r in result.rejections if r.stage == "dedup" and r.rule == "near"]
    assert near, "demo 数据里 wiki.jsonl 有一对近似重复记录，应被 near 判重"
    assert "hamming" in near[0].detail


def test_contaminated_sample_was_rejected(result):
    contaminated = [
        r for r in result.rejections
        if r.stage == "filter" and "contamination" in r.rule
    ]
    assert contaminated, "notes/note1.txt 与评测集完全重合，应被污染规则拦下"
    assert contaminated[0].detail["actual"] > 0.5


def test_low_quality_sample_was_rejected(result):
    lowq = [r for r in result.rejections if r.stage == "filter" and "quality" in r.rule]
    assert lowq, "books.csv 里的 '???' 行应被质量规则拦下"
    assert lowq[0].detail["actual"] == pytest.approx(0.0)


def test_kept_samples_all_pass_filter_rules(result):
    for s in result.dataset.all_samples():
        assert s.metric("quality.score") >= 0.55
        assert s.metric("contamination.ratio") < 0.30


# ----------------------------------------------------------------------
# 可复现性
# ----------------------------------------------------------------------
def test_pipeline_is_deterministic(cfg):
    a = run_pipeline(cfg)
    b = run_pipeline(cfg)
    assert a.version.digest == b.version.digest
    assert a.counters["total_final"] == b.counters["total_final"]

    ids_a = sorted(s.sample_id for s in a.dataset.all_samples())
    ids_b = sorted(s.sample_id for s in b.dataset.all_samples())
    assert ids_a == ids_b


def test_digest_changes_when_config_changes(cfg):
    import copy

    other = copy.deepcopy(cfg)
    other["filter"]["rules"] = [{"metric": "quality.score", "op": ">=", "value": 0.99}]
    assert run_pipeline(other).version.digest != run_pipeline(cfg).version.digest


# ----------------------------------------------------------------------
# 血缘三问
# ----------------------------------------------------------------------
def test_explain_kept_sample_answers_three_questions(result):
    sid = result.dataset.all_samples()[0].sample_id
    expl = explain_sample(result, sid)

    assert expl["found"] is True
    assert expl["status"] == "kept"
    assert expl["split"] in result.dataset.splits
    assert expl["where_from"], "应能回答 Where did this sample come from"
    assert expl["where_from"][0]["kind"] == "source"
    assert expl["what_happened"], "应能回答 What happened to it"
    assert expl["why_here"]["filter_evidence"], "应能回答 Why is it here"


def test_explain_rejected_sample_reports_reason(result):
    rej = result.rejections[0]
    expl = explain_sample(result, rej.sample_id)

    assert expl["found"] is False
    assert expl["status"] == "rejected"
    assert expl["rejections"][0]["stage"] == rej.stage
    assert expl["rejections"][0]["rule"] == rej.rule


def test_explain_unknown_sample(result):
    expl = explain_sample(result, "no:such:id")
    assert expl["found"] is False
    assert expl["status"] == "unknown"
    assert expl["rejections"] == []


def test_lineage_chain_is_consistent(result):
    """每个保留样本的 lineage 必须从 source 节点起步、以 version 节点收尾。"""
    for s in result.dataset.all_samples():
        assert s.lineage[0].startswith("src:")
        assert s.lineage[-1].startswith("ver:")
        for node_id in s.lineage:
            assert node_id in result.dag.nodes


def test_dag_contains_all_source_nodes(result):
    expected = {f"src:{src.source_id}" for src in result.dataset.sources}
    assert expected <= set(result.dag.sources())


# ----------------------------------------------------------------------
# 报告与落盘
# ----------------------------------------------------------------------
def test_report_has_all_sections(result):
    md = build_report(result)
    for section in ("## 一、漏斗", "## 二、来源", "## 三、切分",
                    "## 五、丢弃样本", "## 六、Lineage DAG"):
        assert section in md


def test_write_artifacts_produces_expected_files(result, tmp_path):
    written = write_artifacts(result, tmp_path)

    for key in ("dataset", "rejections", "lineage", "lineage_dot", "version", "report"):
        assert key in written
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "lineage.json").exists()

    # dataset.jsonl 行数 == 最终样本数，且每行都是合法 JSON
    lines = (tmp_path / "dataset.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == result.counters["total_final"]
    for line in lines:
        rec = json.loads(line)
        assert rec["sample_id"] and rec["split"] in result.dataset.splits

    # lineage.json 可被解析且节点数与 DAG 一致
    graph = json.loads((tmp_path / "lineage.json").read_text(encoding="utf-8"))
    assert len(graph["nodes"]) == result.dag.summary()["nodes"]
    assert len(graph["edges"]) == result.dag.summary()["edges"]
