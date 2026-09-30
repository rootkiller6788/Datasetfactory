"""Stage 7 — Filter Engine：按规则表达式过滤。

规则形如::

    { "metric": "quality.score", "op": ">=", "value": 0.55 }

支持 ``>= > <= < == !=``。全部规则必须通过才保留（AND 语义）。

两个关键副作用，都是为了回答「Why is it here?」：
1. 通过者把**每条规则的实际值**写入 ``sample.meta["filter_evidence"]``
2. 未通过者生成 ``Rejection``，记录失败的那条规则与实际值
"""

from __future__ import annotations

import operator
from typing import Any, Callable

from ..lineage import DatasetDAG, LineageNode
from ..schema import Rejection, Sample

OPS: dict[str, Callable[[Any, Any], bool]] = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}


def evaluate(sample: Sample, rule: dict[str, Any]) -> tuple[bool, float | None, str]:
    """返回 (是否通过, 实际值, 说明)。"""
    name = rule["metric"]
    op = rule["op"]
    if op not in OPS:
        raise ValueError(f"unsupported filter op: '{op}'")
    if not sample.has_metric(name):
        return False, None, "metric_missing"
    value = sample.metric(name)
    return bool(OPS[op](value, rule["value"])), value, "ok"


def apply_filter(
    samples: list[Sample], cfg: dict[str, Any], dag: DatasetDAG
) -> tuple[list[Sample], list[Rejection]]:
    rules: list[dict[str, Any]] = cfg.get("rules", [])
    filter_id = cfg.get("filter_id", "default")
    on_fail = cfg.get("on_fail", "drop")

    inputs = sorted({s.head for s in samples if s.head})
    node_id = f"flt:{filter_id}"
    dag.add_node(
        LineageNode(
            node_id=node_id,
            kind="filter",
            label=f"Filter({filter_id})",
            inputs=inputs,
            params={"rules": rules, "on_fail": on_fail},
        )
    )

    kept: list[Sample] = []
    rejected: list[Rejection] = []

    for s in samples:
        failure: dict[str, Any] | None = None
        evidence: list[dict[str, Any]] = []

        for rule in rules:
            passed, value, why = evaluate(s, rule)
            evidence.append(
                {
                    "metric": rule["metric"],
                    "op": rule["op"],
                    "threshold": rule["value"],
                    "actual": value,
                    "passed": passed,
                }
            )
            if not passed and failure is None:
                failure = {
                    "rule": f"{rule['metric']} {rule['op']} {rule['value']}",
                    "metric": rule["metric"],
                    "actual": value,
                    "why": why,
                }

        if failure is not None:
            rejected.append(
                Rejection(
                    sample_id=s.sample_id,
                    source_id=s.source_id,
                    stage="filter",
                    rule=failure["rule"],
                    detail={**failure, "on_fail": on_fail, "evidence": evidence},
                )
            )
            continue

        s.meta.setdefault("filter_evidence", {})[filter_id] = evidence
        s.lineage.append(node_id)
        kept.append(s)

    dag.nodes[node_id].stats = {
        "in": len(samples),
        "out": len(kept),
        "dropped": len(rejected),
    }
    return kept, rejected
