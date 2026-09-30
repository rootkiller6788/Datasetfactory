"""Stage 5 — Quality Score：规则型启发式打分。

四个子分（权重见 ``config.quality.weights``，必须和为 1.0）：

    length          长度分：min(1, chars / length_target)
    content_ratio   内容比：字母 / 数字 / CJK 字符占非空白字符的比例
    line_uniqueness 行唯一性：1 - 重复行占比（单行文本记为 1.0）
    symbol_penalty  符号噪声惩罚：1 - min(1, 符号与控制字符占比 × 5)

``chars < min_chars`` 是硬门槛，直接判 0 分（``hard_fail = "min_chars"``），
避免「短到没有信息量但恰好内容比很高」的样本拿到虚高分。

产物指标：``quality.score`` / ``quality.chars`` / ``quality.content_ratio``
/ ``quality.line_uniqueness`` / ``quality.symbol_penalty``
"""

from __future__ import annotations

import unicodedata
from typing import Any

from ..lineage import DatasetDAG, LineageNode
from ..schema import Sample


def _is_content(ch: str) -> bool:
    """字母 / 数字 / CJK 表意字（category 首字母 L 或 N）。"""
    return unicodedata.category(ch)[0] in ("L", "N")


def _is_noise(ch: str) -> bool:
    """符号 / 控制字符（category 首字母 S 或 C）。"""
    return unicodedata.category(ch)[0] in ("S", "C")


def score_text(text: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """对单条文本打分，返回子分明细。"""
    chars = len(text)
    min_chars = int(cfg.get("min_chars", 0))
    if chars < min_chars:
        return {
            "score": 0.0,
            "chars": chars,
            "length": 0.0,
            "content_ratio": 0.0,
            "line_uniqueness": 0.0,
            "symbol_penalty": 1.0,
            "hard_fail": "min_chars",
        }

    non_space = [c for c in text if not c.isspace()]
    total = len(non_space) or 1
    content_ratio = sum(1 for c in non_space if _is_content(c)) / total
    noise_ratio = sum(1 for c in non_space if _is_noise(c)) / total
    symbol_penalty = 1.0 - min(1.0, noise_ratio * 5.0)

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    line_uniqueness = 1.0 if len(lines) <= 1 else len(set(lines)) / len(lines)

    length_target = max(1, int(cfg.get("length_target", 400)))
    length = min(1.0, chars / length_target)

    w = cfg.get("weights", {})
    score = (
        w.get("length", 0.30) * length
        + w.get("content_ratio", 0.30) * content_ratio
        + w.get("line_uniqueness", 0.25) * line_uniqueness
        + w.get("symbol_penalty", 0.15) * symbol_penalty
    )

    return {
        "score": round(score, 6),
        "chars": chars,
        "length": round(length, 6),
        "content_ratio": round(content_ratio, 6),
        "line_uniqueness": round(line_uniqueness, 6),
        "symbol_penalty": round(symbol_penalty, 6),
        "hard_fail": "",
    }


def score_quality(
    samples: list[Sample], cfg: dict[str, Any], dag: DatasetDAG
) -> list[Sample]:
    inputs = sorted({s.head for s in samples if s.head})
    node_id = "xf:quality"
    dag.add_node(
        LineageNode(
            node_id=node_id,
            kind="transform",
            label="QualityScore",
            inputs=inputs,
            params=dict(cfg),
        )
    )

    for s in samples:
        detail = score_text(s.text, cfg)
        s.set_metric("quality.score", detail["score"], **detail)
        s.set_metric("quality.chars", float(detail["chars"]))
        s.set_metric("quality.content_ratio", detail["content_ratio"])
        s.set_metric("quality.line_uniqueness", detail["line_uniqueness"])
        s.set_metric("quality.symbol_penalty", detail["symbol_penalty"])
        s.lineage.append(node_id)

    scores = [s.metric("quality.score") for s in samples]
    dag.nodes[node_id].stats = {
        "in": len(samples),
        "out": len(samples),
        "mean": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "min": round(min(scores), 4) if scores else 0.0,
        "max": round(max(scores), 4) if scores else 0.0,
    }
    return samples
