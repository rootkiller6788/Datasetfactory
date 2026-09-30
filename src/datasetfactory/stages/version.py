"""Stage 9 — Version：内容寻址的不可变快照。

``digest`` 由全部 ``sample_id`` 排序后拼接再取 sha256 —— 也就是说：

- 内容变了 → digest 变 → 版本不同
- 只是重跑一次 → digest 不变 → 同一个版本

因此 ``digest`` 可以直接当作数据集的「身份证」，用于回归比对与缓存失效判断。
每个 split 在 DAG 上落一个 ``version`` 叶子节点（对应 README 示例里的 ``Train-v4``），
并把该节点 id 追加进这个 split 内每个样本的 ``lineage``，使样本链路以版本收尾：

    src:wiki → xf:normalize → … → spl:train → ver:train:v1

这样 ``explain`` 的「What happened to it」才会落到具体产物上，
样本也才能回答「我属于哪个版本」。
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..lineage import DatasetDAG, LineageNode
from ..schema import Sample, Version, utcnow


def compute_digest(splits: dict[str, list[Sample]]) -> str:
    parts: list[str] = []
    for name in sorted(splits):
        parts.append(f"# {name}")
        parts.extend(sorted(s.sample_id for s in splits[name]))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def make_version(
    splits: dict[str, list[Sample]], cfg: dict[str, Any], dag: DatasetDAG
) -> Version:
    tag = cfg.get("version_tag", "v1")
    parent = cfg.get("parent_version")
    digest = compute_digest(splits)
    counts = {name: len(items) for name, items in splits.items()}

    for name in sorted(splits):
        node_id = f"ver:{name}:{tag}"
        dag.add_node(
            LineageNode(
                node_id=node_id,
                kind="version",
                label=f"{name}-{tag}",
                inputs=[f"spl:{name}"],
                params={"tag": tag, "parent": parent, "digest": digest},
                stats={"samples": len(splits[name])},
            )
        )
        for s in splits[name]:
            s.lineage.append(node_id)

    return Version(
        version_id=tag,
        parent=parent,
        created_at=utcnow(),
        counts=counts,
        digest=digest,
        tag=tag,
    )
