"""Stage 1 — Ingest：把异构来源读成 Canonical Samples。

已实现 kind：``jsonl`` | ``csv`` | ``dir`` | ``txt``
扩展点 kind：``hf``（HuggingFace datasets）| ``web``（抓取 + 正文抽取）

对应 README 架构图里的 ``JSON/CSV/HF / Web/Files``。
每个来源在 DAG 上注册一个 ``source`` 节点，样本的 lineage 从该节点起步。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

from ..config import resolve_path
from ..lineage import DatasetDAG, LineageNode
from ..schema import Sample, Source, short_hash

TEXT_EXTS = {".txt", ".md", ".text"}


def _make_sample(source_id: str, text: str, meta: dict[str, Any]) -> Sample:
    return Sample(
        sample_id=f"{source_id}:{short_hash(text, 12)}",
        text=text,
        source_id=source_id,
        meta=meta,
    )


def load_jsonl(path: Path, text_field: str) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj.get(text_field) or obj.get("content") or "").strip()
            meta = {k: v for k, v in obj.items() if k != text_field}
            meta["_lineno"] = lineno
            rows.append((text, meta))
    return rows


def load_csv(path: Path, text_field: str) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for lineno, obj in enumerate(reader, 2):
            text = str(obj.get(text_field) or "").strip()
            if not text:
                # 兜底：把所有列拼起来，避免静默丢样本
                text = " ".join(str(v) for v in obj.values() if v)
            meta = {k: v for k, v in obj.items() if k != text_field}
            meta["_lineno"] = lineno
            rows.append((text, meta))
    return rows


def load_text_dir(path: Path, text_field: str) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for fp in sorted(path.rglob("*")):
        if not fp.is_file() or fp.suffix.lower() not in TEXT_EXTS:
            continue
        text = fp.read_text(encoding="utf-8").strip()
        rows.append((text, {"file": str(fp.relative_to(path)), "_lineno": 1}))
    return rows


def load_hf(uri: str, text_field: str) -> list[tuple[str, dict[str, Any]]]:
    """扩展点：HuggingFace datasets。需 ``pip install 'datasetfactory[hf]'``。

    注意 ``uri`` 是数据集标识符（如 ``datasets/foo``），不是本地路径。
    """
    raise NotImplementedError(
        "hf loader 为扩展点：安装 datasetfactory[hf] 后在此接入 datasets.load_dataset"
    )


def load_web(uri: str, text_field: str) -> list[tuple[str, dict[str, Any]]]:
    """扩展点：Web 抓取 + 正文抽取，抽取后复用 normalize。``uri`` 为 URL。"""
    raise NotImplementedError("web loader 为扩展点：需接入抓取与正文抽取")


# 本地文件型来源：uri 相对于项目根解析成路径，调用前会校验存在性。
LOCAL_KINDS = {"jsonl", "json", "csv", "dir", "txt"}
# 标识符型来源：uri 是数据集 id / URL，不能当路径解析，直接交给对应 loader。
REMOTE_KINDS = {"hf", "web"}

# 两组 loader 的第一个参数类型不同（Path vs str），故用 Callable[..., ...]。
LOADERS: dict[str, Callable[..., list[tuple[str, dict[str, Any]]]]] = {
    "jsonl": load_jsonl,
    "json": load_jsonl,
    "csv": load_csv,
    "dir": load_text_dir,
    "txt": load_text_dir,
    "hf": load_hf,
    "web": load_web,
}


def ingest(
    cfg: dict[str, Any], dag: DatasetDAG
) -> tuple[list[Sample], list[Source]]:
    """读入所有来源，返回 (samples, sources)。"""
    samples: list[Sample] = []
    sources: list[Source] = []

    for spec in cfg["sources"]:
        sid = spec["source_id"]
        kind = spec["kind"]
        if kind not in LOADERS:
            raise ValueError(
                f"unsupported source kind: '{kind}' (source '{sid}')；"
                f"可用：{', '.join(sorted(LOADERS))}"
            )

        raw_uri = spec["uri"]
        text_field = spec.get("text_field", "text")

        if kind in REMOTE_KINDS:
            # 标识符型来源没有本地文件可校验，直接交给 loader（未实现则抛 NotImplementedError）
            rows = LOADERS[kind](raw_uri, text_field)
        else:
            path = resolve_path(cfg["_root"], raw_uri)
            if not path.exists():
                raise FileNotFoundError(f"source '{sid}' 找不到文件：{path}")
            rows = LOADERS[kind](path, text_field)

        node_id = f"src:{sid}"
        dag.add_node(
            LineageNode(
                node_id=node_id,
                kind="source",
                label=f"Source({sid})",
                params={
                    "uri": spec["uri"],
                    "kind": kind,
                    "license": spec.get("license", "unknown"),
                    "text_field": text_field,
                },
                stats={"samples": len(rows)},
            )
        )
        sources.append(
            Source(
                source_id=sid,
                uri=spec["uri"],
                kind=kind,
                license=spec.get("license", "unknown"),
                meta={"text_field": text_field, "samples": len(rows)},
            )
        )

        for text, meta in rows:
            s = _make_sample(sid, text, meta)
            s.lineage.append(node_id)
            samples.append(s)

    return samples, sources
