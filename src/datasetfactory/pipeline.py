"""流水线编排：把九个 stage 串成一次可复现的运行。

    Raw Sources → Ingest → Normalize → Fingerprint → Deduplicate
    → Quality Score → Contamination Check → Filter → Split → Version
    → Dataset Artifact

顺序不可随意调整：``Filter`` 的规则通常引用 ``quality.score`` 与
``contamination.ratio``，因此这两步必须在 ``Filter`` 之前完成。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .lineage import DatasetDAG
from .schema import CODE_VERSION, Dataset, Rejection, Sample, Transform
from .stages import contamination as contamination_stage
from .stages import dedup as dedup_stage
from .stages import filter as filter_stage
from .stages import fingerprint as fingerprint_stage
from .stages import ingest as ingest_stage
from .stages import normalize as normalize_stage
from .stages import quality as quality_stage
from .stages import split as split_stage
from .stages import version as version_stage


@dataclass
class PipelineResult:
    """一次运行的完整产物。"""

    dataset: Dataset
    dag: DatasetDAG
    rejections: list[Rejection] = field(default_factory=list)
    counters: dict[str, Any] = field(default_factory=dict)
    cfg: dict[str, Any] = field(default_factory=dict)

    # -- 便捷查询 ----------------------------------------------------
    def find(self, sample_id: str) -> tuple[str, Sample] | None:
        return self.dataset.find(sample_id)

    def rejections_for(self, sample_id: str) -> list[Rejection]:
        return [r for r in self.rejections if r.sample_id == sample_id]

    def known_ids(self) -> list[str]:
        ids = [s.sample_id for s in self.dataset.all_samples()]
        ids.extend(r.sample_id for r in self.rejections)
        return sorted(set(ids))

    @property
    def version(self):
        return self.dataset.version


def run_pipeline(cfg: dict[str, Any]) -> PipelineResult:
    """执行完整流水线。``cfg`` 由 :func:`datasetfactory.config.load_config` 产出。"""
    dag = DatasetDAG()
    rejections: list[Rejection] = []
    counters: dict[str, Any] = {}
    transforms: list[Transform] = []

    # 1) Ingest -------------------------------------------------------
    samples, sources = ingest_stage.ingest(cfg, dag)
    counters["raw"] = len(samples)

    # 2) Normalize ----------------------------------------------------
    samples = normalize_stage.normalize(samples, cfg["normalize"], dag)
    counters["normalized"] = len(samples)
    transforms.append(
        Transform("xf:normalize", "Normalize", dict(cfg["normalize"]), CODE_VERSION)
    )

    # 3) Fingerprint --------------------------------------------------
    samples = fingerprint_stage.fingerprint(samples, cfg["fingerprint"], dag)
    counters["fingerprinted"] = len(samples)
    transforms.append(
        Transform("xf:fingerprint", "Fingerprint", dict(cfg["fingerprint"]), CODE_VERSION)
    )

    # 4) Deduplicate --------------------------------------------------
    samples, dropped = dedup_stage.deduplicate(samples, cfg["dedup"], dag)
    rejections.extend(dropped)
    counters["deduped"] = len(samples)
    transforms.append(Transform("xf:dedup", "Deduplicate", dict(cfg["dedup"]), CODE_VERSION))

    # 5) Quality Score ------------------------------------------------
    samples = quality_stage.score_quality(samples, cfg["quality"], dag)
    counters["quality_scored"] = len(samples)
    transforms.append(Transform("xf:quality", "QualityScore", dict(cfg["quality"]), CODE_VERSION))

    # 6) Contamination Check ------------------------------------------
    samples = contamination_stage.check_contamination(
        samples, cfg["contamination"], dag, cfg["_root"]
    )
    counters["contamination_checked"] = len(samples)
    transforms.append(
        Transform(
            "xf:contamination", "ContaminationCheck", dict(cfg["contamination"]), CODE_VERSION
        )
    )

    # 7) Filter -------------------------------------------------------
    samples, dropped = filter_stage.apply_filter(samples, cfg["filter"], dag)
    rejections.extend(dropped)
    counters["filtered"] = len(samples)
    transforms.append(Transform("flt:main", "Filter", dict(cfg["filter"]), CODE_VERSION))

    # 8) Split --------------------------------------------------------
    splits = split_stage.split_dataset(samples, cfg["split"], dag)
    counters["splits"] = {name: len(items) for name, items in splits.items()}

    # 9) Version ------------------------------------------------------
    ver = version_stage.make_version(splits, cfg, dag)
    counters["final"] = dict(ver.counts)
    counters["total_final"] = sum(ver.counts.values())
    counters["rejected"] = len(rejections)
    transforms.append(Transform("ver", "Version", {"tag": ver.tag}, CODE_VERSION))

    dataset = Dataset(
        dataset_id=cfg.get("pipeline_id", "datasetfactory"),
        version=ver,
        splits=splits,
        sources=sources,
        transforms=transforms,
    )

    return PipelineResult(
        dataset=dataset, dag=dag, rejections=rejections, counters=counters, cfg=cfg
    )
