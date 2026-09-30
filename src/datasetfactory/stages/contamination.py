"""Stage 6 — Contamination Check：与评测集做 n-gram 重合度检测。

做法：先把参考集（评测集）全部切成 n-gram 建索引，再算每条样本的
n-gram 有多少比例出现在参考集里。

    contamination.ratio = |ngrams(sample) ∩ ngrams(reference)| / |ngrams(sample)|

ratio 越接近 1，说明该样本越可能直接来自评测集（数据污染）。
默认 n=8：短于 8 个字符的巧合重叠基本可忽略，长于 8 的重合才有信号。

必须排在 Filter 之前，因为 filter 规则通常会引用 ``contamination.ratio``。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import resolve_path
from ..lineage import DatasetDAG, LineageNode
from ..schema import Sample


def ngrams(text: str, n: int) -> set[str]:
    """去空白后的字符 n-gram 集合。"""
    compact = "".join(text.split())
    if not compact:
        return set()
    if len(compact) < n:
        return {compact}
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def load_reference(cfg: dict[str, Any], root: str | Path) -> tuple[set[str], dict[str, Any]]:
    """加载参考集并切成 n-gram。支持 .jsonl / .json / .txt。"""
    rel = cfg.get("reference")
    if not rel:
        return set(), {"path": None, "documents": 0, "ngrams": 0}

    path: Path = resolve_path(root, rel)
    if not path.exists():
        raise FileNotFoundError(f"contamination reference not found: {path}")

    text_field = cfg.get("text_field", "text")
    texts: list[str] = []

    if path.suffix.lower() in (".jsonl", ".json"):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = str(obj.get(text_field) or obj.get("content") or "").strip()
                if text:
                    texts.append(text)
    else:
        texts.append(path.read_text(encoding="utf-8"))

    size = int(cfg.get("ngram", 8))
    pool: set[str] = set()
    for text in texts:
        pool |= ngrams(text, size)

    return pool, {
        "path": str(path),
        "documents": len(texts),
        "ngrams": len(pool),
        "ngram": size,
    }


def check_contamination(
    samples: list[Sample],
    cfg: dict[str, Any],
    dag: DatasetDAG,
    root: str | Path = ".",
) -> list[Sample]:
    pool, ref_stats = load_reference(cfg, root)
    size = int(cfg.get("ngram", 8))
    threshold = float(cfg.get("threshold", 0.30))

    inputs = sorted({s.head for s in samples if s.head})
    node_id = "xf:contamination"
    dag.add_node(
        LineageNode(
            node_id=node_id,
            kind="transform",
            label="ContaminationCheck",
            inputs=inputs,
            params={"reference": ref_stats.get("path"), "ngram": size, "threshold": threshold},
        )
    )

    flagged = 0
    for s in samples:
        grams = ngrams(s.text, size)
        if not grams or not pool:
            ratio = 0.0
            overlap = 0
        else:
            overlap = len(grams & pool)
            ratio = overlap / len(grams)

        s.set_metric(
            "contamination.ratio",
            round(ratio, 6),
            overlap=overlap,
            total=len(grams),
            ngram=size,
            reference=ref_stats.get("path"),
        )
        if ratio >= threshold:
            flagged += 1
        s.lineage.append(node_id)

    dag.nodes[node_id].stats = {
        "in": len(samples),
        "out": len(samples),
        "reference_docs": ref_stats.get("documents", 0),
        "flagged": flagged,
    }
    return samples
