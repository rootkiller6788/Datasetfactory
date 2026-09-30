"""Stage 4 — Deduplicate：Exact（sha256）+ Near（SimHash 汉明距离）。

- Exact：归一化文本的 sha256 完全相同 → 判重
- Near：与已保留样本的 SimHash 汉明距离 ≤ 阈值 → 判重

**阈值怎么定（重要）**：SimHash 的汉明距离强烈依赖文本长度 ——
同样一处改动，文本越短，翻转的位数越多。实测（64bit SimHash + 字符 5-shingle）：

    187 shingle 的长文本：删 1 字 → 2   改 1 词 → 6   改 2 词 → 9   改 3 词 → 13
     41 shingle 的短文本：改 2 个词     → 13
    无关文本（任意长度） → 26 ~ 35

也就是说：不存在一个对长短文本都合适的绝对阈值。短文本（shingle 数很少）
的指纹本身就接近随机，此时任何阈值都会同时出现漏检与误杀。

因此本实现把默认值取保守的 ``8``：**漏检一个近似重复只是少删一条，
误合并却会静默丢掉一条真实数据**，后者更贵。数据以长文本为主时可放宽到
10~12（无关文本在 26 以上，仍有充足余量）；短文本占比高时应在
``_find_near_duplicate`` 之前加一道最小长度门槛，直接跳过指纹不可靠的样本。

规模说明：near 判定当前是「与已保留集合逐一比较」（O(n·k)），
在十万级以内可用。更大规模应换成 SimHash 分段索引（banding）或 MinHash LSH，
接口不变，只需替换 ``_find_near_duplicate`` 的实现。
"""

from __future__ import annotations

from typing import Any

from ..lineage import DatasetDAG, LineageNode
from ..schema import Rejection, Sample, sha256_text
from .fingerprint import hamming


def _find_near_duplicate(
    sample: Sample, kept: list[Sample], threshold: int
) -> tuple[Sample, int] | None:
    for candidate in kept:
        distance = hamming(sample.fingerprint, candidate.fingerprint)
        if distance <= threshold:
            return candidate, distance
    return None


def deduplicate(
    samples: list[Sample], cfg: dict[str, Any], dag: DatasetDAG
) -> tuple[list[Sample], list[Rejection]]:
    exact_enabled = bool(cfg.get("exact", True))
    near_cfg = cfg.get("near") or {}
    near_enabled = bool(near_cfg.get("enabled", False))
    threshold = int(near_cfg.get("hamming_threshold", 8))

    inputs = sorted({s.head for s in samples if s.head})
    node_id = "xf:dedup"
    dag.add_node(
        LineageNode(
            node_id=node_id,
            kind="transform",
            label="Deduplicate",
            inputs=inputs,
            params={"exact": exact_enabled, "near": near_cfg},
        )
    )

    kept: list[Sample] = []
    rejected: list[Rejection] = []
    seen_digest: dict[str, str] = {}

    for s in samples:
        digest = sha256_text(s.text)

        if exact_enabled and digest in seen_digest:
            rejected.append(
                Rejection(
                    sample_id=s.sample_id,
                    source_id=s.source_id,
                    stage="dedup",
                    rule="exact",
                    detail={"duplicate_of": seen_digest[digest], "digest": digest[:16]},
                )
            )
            continue

        if near_enabled:
            hit = _find_near_duplicate(s, kept, threshold)
            if hit is not None:
                candidate, distance = hit
                rejected.append(
                    Rejection(
                        sample_id=s.sample_id,
                        source_id=s.source_id,
                        stage="dedup",
                        rule="near",
                        detail={
                            "duplicate_of": candidate.sample_id,
                            "hamming": distance,
                            "threshold": threshold,
                        },
                    )
                )
                continue

        seen_digest[digest] = s.sample_id
        s.meta["exact_digest"] = digest
        s.lineage.append(node_id)
        kept.append(s)

    dag.nodes[node_id].stats = {
        "in": len(samples),
        "out": len(kept),
        "exact_dropped": sum(1 for r in rejected if r.rule == "exact"),
        "near_dropped": sum(1 for r in rejected if r.rule == "near"),
    }
    return kept, rejected
