"""DatasetFactory — 数据集工厂。

对应 README 的九段流水线：

    Raw Sources → Ingest → Normalize → Fingerprint → Deduplicate
    → Quality Score → Contamination Check → Filter → Split → Version
    → Dataset Artifact

核心链路零第三方依赖（落地约束：4060 8G 单机可跑）。
"""

from .config import load_config
from .lineage import DatasetDAG, LineageNode
from .pipeline import PipelineResult, run_pipeline
from .schema import (
    CODE_VERSION,
    Dataset,
    Filter,
    Metric,
    Rejection,
    Sample,
    Source,
    Split,
    Transform,
    Version,
)

__version__ = CODE_VERSION

__all__ = [
    "CODE_VERSION",
    "Dataset",
    "DatasetDAG",
    "Filter",
    "LineageNode",
    "Metric",
    "PipelineResult",
    "Rejection",
    "Sample",
    "Source",
    "Split",
    "Transform",
    "Version",
    "load_config",
    "run_pipeline",
    "__version__",
]
