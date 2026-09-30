"""九段流水线。

每个 stage 都是 ``(samples, cfg, dag) -> ...`` 形式的纯函数，
并且都会在 DAG 上注册一个 ``LineageNode``、把该节点 id 追加进存活样本的 ``lineage``。
这样任何样本最终都能回溯出完整血缘。
"""

from . import (
    contamination,
    dedup,
    filter,
    fingerprint,
    ingest,
    normalize,
    quality,
    split,
    version,
)

__all__ = [
    "contamination",
    "dedup",
    "filter",
    "fingerprint",
    "ingest",
    "normalize",
    "quality",
    "split",
    "version",
]
