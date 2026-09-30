"""配置加载与默认值合并。

配置为纯 JSON（避免引入 PyYAML），约定放在 ``<root>/configs/*.json``，
因此 ``_root`` 取配置文件的上两级目录，配置里的相对路径都相对 ``_root`` 解析。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "pipeline_id": "datasetfactory",
    "version_tag": "v1",
    "parent_version": None,
    "sources": [],
    "normalize": {
        "unicode_form": "NFKC",
        "strip_html": True,
        "collapse_whitespace": True,
        "drop_empty": True,
    },
    "fingerprint": {"algo": "simhash", "bits": 64, "shingle_size": 5},
    "dedup": {"exact": True, "near": {"enabled": True, "hamming_threshold": 8}},
    "quality": {
        "min_chars": 40,
        "length_target": 400,
        "weights": {
            "length": 0.30,
            "content_ratio": 0.30,
            "line_uniqueness": 0.25,
            "symbol_penalty": 0.15,
        },
    },
    "contamination": {
        "reference": None,
        "text_field": "text",
        "ngram": 8,
        "threshold": 0.30,
    },
    "filter": {"filter_id": "default", "on_fail": "drop", "rules": []},
    "split": {
        "name": "main",
        "seed": 42,
        "ratios": {"train": 0.8, "valid": 0.1, "test": 0.1},
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """递归合并：override 覆盖 base，dict 逐层深入，其余整体替换。"""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 JSON 配置并与 DEFAULTS 合并，附带 ``_config_path`` / ``_root``。"""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    cfg = deep_merge(DEFAULTS, raw)
    cfg["_config_path"] = str(p)
    cfg["_root"] = str(p.parent.parent)
    validate(cfg)
    return cfg


def resolve_path(root: str | Path, rel: str | Path) -> Path:
    """把配置里的相对路径解析为绝对路径（相对 ``root``）。"""
    p = Path(rel)
    return p if p.is_absolute() else Path(root) / p


def validate(cfg: dict[str, Any]) -> None:
    if not cfg.get("sources"):
        raise ValueError("config.sources 不能为空")

    seen: set[str] = set()
    for spec in cfg["sources"]:
        for field in ("source_id", "uri", "kind"):
            if not spec.get(field):
                raise ValueError(f"source 缺少字段 '{field}': {spec}")
        if spec["source_id"] in seen:
            raise ValueError(f"source_id 重复: {spec['source_id']}")
        seen.add(spec["source_id"])

    ratios = cfg["split"]["ratios"]
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split.ratios 之和必须为 1.0，当前为 {total}")

    weights = cfg["quality"]["weights"]
    wsum = sum(weights.values())
    if abs(wsum - 1.0) > 1e-6:
        raise ValueError(f"quality.weights 之和必须为 1.0，当前为 {wsum}")

    # 延迟导入：stages/__init__ 会拉起 contamination，而它要
    # ``from ..config import resolve_path``——模块级导入会成环。
    # 此处的职责是让非法 op 在**配置加载期**就报错，而不是等跑到
    # Filter 阶段（前面六个 stage 全白跑）才在 evaluate() 里抛。
    from .stages.filter import OPS

    for rule in cfg["filter"]["rules"]:
        for field in ("metric", "op", "value"):
            if field not in rule:
                raise ValueError(f"filter rule 缺少字段 '{field}': {rule}")
        if rule["op"] not in OPS:
            raise ValueError(
                f"filter rule op 不支持: {rule['op']!r}；"
                f"可用：{', '.join(sorted(OPS))}"
            )
