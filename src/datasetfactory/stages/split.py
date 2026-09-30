"""Stage 8 — Split：按 ``sample_id`` 哈希做确定性切分。

用哈希而非随机数，好处是：
- 同一份数据重复运行，切分结果完全一致（可复现）
- 新增样本不会打乱已有样本的归属（增量友好）

**小样本注意**：样本数很少时，哈希切分会天然出现某一路为空
（例如 13 条样本、15% 配额，落空概率约 12%）。这不是 bug，
而是配额比例在有限样本上的正常波动。若需要每路非空，
请增大样本量，或换一个 ``split.seed`` 重试 —— 报告里会对空 split 给出提示。

每个 split 在 DAG 上注册一个独立节点，因此血缘图会自然呈现扇出结构。
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..lineage import DatasetDAG, LineageNode
from ..schema import Sample


def bucket(sample_id: str, seed: int) -> float:
    """把 sample_id 映射到 [0, 1)。"""
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_split(sample_id: str, ratios: dict[str, float], seed: int) -> str:
    """按累积比例区间落桶。"""
    position = bucket(sample_id, seed)
    cumulative = 0.0
    last = next(iter(ratios))
    for name, ratio in ratios.items():
        cumulative += ratio
        last = name
        if position < cumulative:
            return name
    return last


def split_dataset(
    samples: list[Sample], cfg: dict[str, Any], dag: DatasetDAG
) -> dict[str, list[Sample]]:
    name = cfg.get("name", "main")
    ratios: dict[str, float] = cfg["ratios"]
    seed = int(cfg.get("seed", 42))

    splits: dict[str, list[Sample]] = {key: [] for key in ratios}
    for s in samples:
        target = assign_split(s.sample_id, ratios, seed)
        s.meta["split"] = target
        s.meta["split_seed"] = seed
        splits[target].append(s)

    inputs = sorted({s.head for s in samples if s.head})
    for split_name in ratios:
        node_id = f"spl:{split_name}"
        dag.add_node(
            LineageNode(
                node_id=node_id,
                kind="split",
                label=f"Split/{split_name}",
                inputs=inputs,
                params={"ratios": ratios, "seed": seed, "name": name},
                stats={"samples": len(splits[split_name])},
            )
        )

    for s in samples:
        s.lineage.append(f"spl:{s.meta['split']}")

    return splits
