"""DatasetFactory 核心对象。

对应 README「六、核心对象」：
Dataset / Sample / Source / Transform / Metric / Filter / Split / Version

设计原则
--------
1. 全部为 ``dataclass``，可直接 ``asdict()`` 序列化到 ``lineage.json`` / ``dataset.jsonl``
2. 核心链路零第三方依赖（落地约束：4060 8G 单机可跑）
3. ``sample_id`` 由「来源 + 归一化文本」决定，跨运行稳定，保证结果可复现
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

CODE_VERSION = "0.1.0"


def utcnow() -> str:
    """ISO-8601 UTC 时间戳（秒级）。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    """文本的 sha256 十六进制摘要。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text: str, n: int = 12) -> str:
    """取 sha256 前 n 位，用作短标识。"""
    return sha256_text(text)[:n]


@dataclass
class Source:
    """原始来源：路径 / 授权 / 抓取时间。"""

    source_id: str
    uri: str
    kind: str
    license: str = "unknown"
    fetched_at: str = field(default_factory=utcnow)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Transform:
    """一次可复现的处理步骤（含参数）。"""

    transform_id: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    code_version: str = CODE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Metric:
    """质量 / 去重 / 污染的度量定义。"""

    name: str
    value: float
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Sample:
    """最小数据单元，携带 lineage 引用。"""

    sample_id: str
    text: str
    source_id: str
    fingerprint: int = 0
    lineage: list[str] = field(default_factory=list)
    metrics: dict[str, Metric] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- metric 读写 -------------------------------------------------
    def set_metric(self, name: str, value: float, **detail: Any) -> None:
        self.metrics[name] = Metric(name=name, value=float(value), detail=dict(detail))

    def metric(self, name: str, default: float = 0.0) -> float:
        m = self.metrics.get(name)
        return m.value if m is not None else default

    def has_metric(self, name: str) -> bool:
        return name in self.metrics

    # -- lineage -----------------------------------------------------
    @property
    def head(self) -> str:
        """当前所处的 lineage 节点（链路末端）。"""
        return self.lineage[-1] if self.lineage else ""

    @property
    def chars(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Filter:
    """基于 Metric 的保留 / 丢弃规则。"""

    filter_id: str
    rules: list[dict[str, Any]] = field(default_factory=list)
    on_fail: str = "drop"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Split:
    """train / valid / test 切分策略。"""

    name: str
    ratios: dict[str, float] = field(default_factory=dict)
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Version:
    """不可变的数据集快照。"""

    version_id: str
    parent: str | None
    created_at: str
    counts: dict[str, int] = field(default_factory=dict)
    digest: str = ""
    tag: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Rejection:
    """被丢弃的样本及原因 —— 回答 sample 的「Why NOT here」。"""

    sample_id: str
    source_id: str
    stage: str
    rule: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Dataset:
    """一次产物的容器，绑定版本与 split。"""

    dataset_id: str
    version: Version
    splits: dict[str, list[Sample]] = field(default_factory=dict)
    sources: list[Source] = field(default_factory=list)
    transforms: list[Transform] = field(default_factory=list)

    def all_samples(self) -> list[Sample]:
        out: list[Sample] = []
        for name in sorted(self.splits):
            out.extend(self.splits[name])
        return out

    def size(self) -> int:
        return sum(len(v) for v in self.splits.values())

    def find(self, sample_id: str) -> tuple[str, Sample] | None:
        """按 sample_id 查找，返回 (split_name, sample)。"""
        for name in sorted(self.splits):
            for s in self.splits[name]:
                if s.sample_id == sample_id:
                    return name, s
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version.to_dict(),
            "splits": {k: [s.to_dict() for s in v] for k, v in self.splits.items()},
            "sources": [s.to_dict() for s in self.sources],
            "transforms": [t.to_dict() for t in self.transforms],
        }
