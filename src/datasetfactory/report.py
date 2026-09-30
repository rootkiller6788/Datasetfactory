"""报告与血缘解释。

README「五、Lineage DAG」要求任何 sample 都能回答三个问题：

    Where did this sample come from?   → ``explain_sample()["where_from"]``
    What happened to it?               → ``explain_sample()["what_happened"]``
    Why is it here?                    → ``explain_sample()["why_here"]``

本模块提供这三个答案，以及把整次运行落盘为可审计的 Artifact。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .pipeline import PipelineResult
from .schema import Sample


# ----------------------------------------------------------------------
# 血缘解释
# ----------------------------------------------------------------------
def explain_sample(result: PipelineResult, sample_id: str) -> dict[str, Any]:
    """回答某个 sample 的三个问题；若已被丢弃，则给出丢弃原因。"""
    hit = result.dataset.find(sample_id)
    rejections = result.rejections_for(sample_id)

    if hit is None:
        return {
            "found": False,
            "sample_id": sample_id,
            "status": "rejected" if rejections else "unknown",
            "where_from": [],
            "what_happened": [],
            "why_here": {},
            "rejections": [r.to_dict() for r in rejections],
        }

    split, sample = hit
    nodes = [result.dag.nodes[n] for n in sample.lineage if n in result.dag.nodes]

    return {
        "found": True,
        "sample_id": sample_id,
        "status": "kept",
        "split": split,
        "chars": sample.chars,
        "where_from": [n.to_dict() for n in nodes if n.kind == "source"],
        "what_happened": [n.to_dict() for n in nodes if n.kind != "source"],
        "why_here": {
            "filter_evidence": sample.meta.get("filter_evidence", {}),
            "metrics": {k: v.value for k, v in sorted(sample.metrics.items())},
        },
        "path": list(sample.lineage),
    }


def render_explanation(expl: dict[str, Any]) -> str:
    """把 explain_sample 的结果渲染成终端友好的文本。"""
    sid = expl["sample_id"]
    status = expl["status"]
    label = {
        "kept": "kept（在数据集中）",
        "rejected": "rejected（已被丢弃，不在数据集中）",
        "unknown": "unknown（本次运行未见过这个 id）",
    }.get(status, status)

    lines: list[str] = [f"sample: {sid}", f"status: {label}"]

    if not expl["found"]:
        for r in expl.get("rejections", []):
            lines.append(f"  ✗ 在 [{r['stage']}] 阶段被丢弃：{r['rule']}")
            detail = r.get("detail", {})
            if detail.get("actual") is not None:
                lines.append(f"    实际值 = {detail['actual']}")
            if detail.get("duplicate_of"):
                lines.append(f"    重复于 = {detail['duplicate_of']}")
        return "\n".join(lines)

    lines.append(f"split : {expl['split']}   chars: {expl['chars']}")
    lines.append("")
    lines.append("Where did this sample come from?")
    for n in expl["where_from"]:
        p = n.get("params", {})
        lines.append(f"  ← {n['label']}  uri={p.get('uri')}  license={p.get('license')}")

    lines.append("")
    lines.append("What happened to it?")
    for n in expl["what_happened"]:
        stat = "  ".join(f"{k}={v}" for k, v in (n.get("stats") or {}).items())
        lines.append(f"  → {n['label']} [{n['kind']}]{('  ' + stat) if stat else ''}")

    lines.append("")
    lines.append("Why is it here?")
    for fid, evidence in expl["why_here"]["filter_evidence"].items():
        lines.append(f"  filter={fid}")
        for e in evidence:
            mark = "✓" if e["passed"] else "✗"
            lines.append(
                f"    {mark} {e['metric']} {e['op']} {e['threshold']}  (实际 {e['actual']})"
            )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def _funnel_rows(result: PipelineResult) -> list[tuple[str, str, str, str]]:
    c = result.counters
    raw = c.get("raw", 0)
    normalized = c.get("normalized", 0)
    fingerprinted = c.get("fingerprinted", 0)
    deduped = c.get("deduped", 0)
    scored = c.get("quality_scored", 0)
    contam = c.get("contamination_checked", 0)
    filtered = c.get("filtered", 0)
    total_final = c.get("total_final", 0)

    def diff(a: int, b: int) -> str:
        d = a - b
        return str(d) if d else "—"

    return [
        ("Ingest", "—", str(raw), "—"),
        ("Normalize", str(raw), str(normalized), diff(raw, normalized)),
        ("Fingerprint", str(normalized), str(fingerprinted), "—"),
        ("Deduplicate", str(fingerprinted), str(deduped), diff(fingerprinted, deduped)),
        ("Quality Score", str(deduped), str(scored), "—"),
        ("Contamination Check", str(scored), str(contam), "—"),
        ("Filter", str(contam), str(filtered), diff(contam, filtered)),
        ("Split", str(filtered), str(total_final), "—"),
    ]


def _rejection_detail(r: Any) -> str:
    """把丢弃原因渲染成一行人类可读的说明。"""
    d = r.detail
    if r.stage == "dedup":
        parts = [f"duplicate_of=`{d.get('duplicate_of')}`"]
        if "hamming" in d:
            parts.append(f"hamming={d['hamming']} ≤ {d.get('threshold')}")
        else:
            parts.append("内容 sha256 完全相同")
        return "，".join(parts)
    if r.stage == "filter":
        actual = d.get("actual")
        actual_s = "指标缺失" if actual is None else f"{actual:.4f}"
        return f"实际 {actual_s}，未满足 `{r.rule}`"
    return json.dumps(d, ensure_ascii=False)


def build_report(result: PipelineResult) -> str:
    """生成 Markdown 运行报告。"""
    ds = result.dataset
    ver = ds.version

    lines: list[str] = []
    lines.append("# DatasetFactory 运行报告")
    lines.append("")
    lines.append(f"- pipeline: `{ds.dataset_id}`")
    lines.append(f"- 版本: `{ver.version_id}`  (digest `{ver.digest[:16]}…`)")
    lines.append(f"- 生成时间: {ver.created_at}")
    lines.append(f"- 父版本: {ver.parent or '—'}")
    lines.append("")

    # 漏斗
    lines.append("## 一、漏斗")
    lines.append("")
    lines.append("| 阶段 | 输入 | 输出 | 丢弃 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for name, a, b, d in _funnel_rows(result):
        lines.append(f"| {name} | {a} | {b} | {d} |")
    lines.append("")
    lines.append(
        f"**总计**：原始 {result.counters.get('raw', 0)} 条 → "
        f"最终 {result.counters.get('total_final', 0)} 条，"
        f"累计丢弃 {result.counters.get('rejected', 0)} 条。"
    )
    lines.append("")

    # 来源
    lines.append("## 二、来源")
    lines.append("")
    lines.append("| source_id | kind | uri | license | 样本数 |")
    lines.append("| --- | --- | --- | --- | ---: |")
    for s in ds.sources:
        lines.append(
            f"| {s.source_id} | {s.kind} | `{s.uri}` | {s.license} | "
            f"{s.meta.get('samples', '—')} |"
        )
    lines.append("")

    # 切分
    lines.append("## 三、切分")
    lines.append("")
    lines.append("| split | 样本数 | 占比 | 目标占比 |")
    lines.append("| --- | ---: | ---: | ---: |")
    total = sum(len(v) for v in ds.splits.values()) or 1
    targets = result.cfg.get("split", {}).get("ratios", {})
    empty: list[str] = []
    for name in sorted(ds.splits):
        n = len(ds.splits[name])
        if n == 0:
            empty.append(name)
        target = targets.get(name)
        target_s = f"{target:.0%}" if isinstance(target, (int, float)) else "—"
        lines.append(f"| {name} | {n} | {n / total:.1%} | {target_s} |")
    lines.append("")
    if empty:
        lines.append(
            f"> ⚠️ split {', '.join('`' + e + '`' for e in empty)} 为空。"
            "小样本下哈希切分会自然出现空集，可增大样本量或更换 `split.seed`。"
        )
        lines.append("")

    # 质量分布
    kept = ds.all_samples()
    if kept:
        scores = sorted(s.metric("quality.score") for s in kept)
        ratios = sorted(s.metric("contamination.ratio") for s in kept)
        mid = scores[len(scores) // 2]
        lines.append("## 四、保留样本的质量分布")
        lines.append("")
        lines.append("| 指标 | 最小 | 中位 | 最大 |")
        lines.append("| --- | ---: | ---: | ---: |")
        lines.append(
            f"| quality.score | {scores[0]:.3f} | {mid:.3f} | {scores[-1]:.3f} |"
        )
        lines.append(
            f"| contamination.ratio | {ratios[0]:.3f} | "
            f"{ratios[len(ratios) // 2]:.3f} | {ratios[-1]:.3f} |"
        )
        lines.append("")

    # 丢弃
    lines.append("## 五、丢弃样本")
    lines.append("")
    if result.rejections:
        lines.append("| sample_id | 阶段 | 规则 | 详情 |")
        lines.append("| --- | --- | --- | --- |")
        for r in result.rejections:
            lines.append(
                f"| `{r.sample_id}` | {r.stage} | `{r.rule}` | {_rejection_detail(r)} |"
            )
    else:
        lines.append("_本次运行没有样本被丢弃。_")
    lines.append("")

    # Lineage DAG
    lines.append("## 六、Lineage DAG")
    lines.append("")
    lines.append(f"节点 {result.dag.summary()['nodes']} 个，"
                 f"边 {result.dag.summary()['edges']} 条。")
    lines.append("")
    lines.append("```")
    lines.append(result.dag.to_ascii())
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# 落盘
# ----------------------------------------------------------------------
def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _sample_record(split: str, s: Sample) -> dict[str, Any]:
    return {
        "sample_id": s.sample_id,
        "split": split,
        "source_id": s.source_id,
        "chars": s.chars,
        "fingerprint": s.fingerprint,
        "metrics": {k: v.value for k, v in sorted(s.metrics.items())},
        "filter_evidence": s.meta.get("filter_evidence", {}),
        "lineage": s.lineage,
        "text": s.text,
    }


def write_artifacts(result: PipelineResult, out_dir: str | Path) -> dict[str, str]:
    """把一次运行的全部产物落盘，返回 {名称: 路径}。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    ds = result.dataset

    # dataset.jsonl（全量，带 split 字段）
    all_records = [
        _sample_record(name, s) for name in sorted(ds.splits) for s in ds.splits[name]
    ]
    p = out / "dataset.jsonl"
    _write_jsonl(p, all_records)
    written["dataset"] = str(p)

    # 分 split 文件
    for name in sorted(ds.splits):
        p = out / "splits" / f"{name}.jsonl"
        _write_jsonl(p, [_sample_record(name, s) for s in ds.splits[name]])
        written[f"split:{name}"] = str(p)

    # 丢弃记录
    p = out / "rejections.jsonl"
    _write_jsonl(p, [r.to_dict() for r in result.rejections])
    written["rejections"] = str(p)

    # 血缘图
    p = out / "lineage.json"
    result.dag.save(p)
    written["lineage"] = str(p)

    p = out / "lineage.dot"
    p.write_text(result.dag.to_dot(), encoding="utf-8")
    written["lineage_dot"] = str(p)

    # 版本
    p = out / "version.json"
    p.write_text(
        json.dumps(ds.version.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written["version"] = str(p)

    # 报告
    p = out / "report.md"
    p.write_text(build_report(result), encoding="utf-8")
    written["report"] = str(p)

    return written
