"""Stage 2 — Normalize：Unicode NFKC、去 HTML、压缩空白、清零宽字符。

归一化后重算 ``sample_id``（来源 + 归一化文本），因此：
- 同一来源下内容完全相同的两条记录会得到同一个 id（配合 dedup 的 exact 判定）
- 换个运行环境、换个顺序，id 依然稳定
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

from ..lineage import DatasetDAG, LineageNode
from ..schema import Sample, sha256_text, short_hash

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"[ \t\u00a0\u3000]+")
MULTI_NL_RE = re.compile(r"\n{3,}")
ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u2060"), None)


def normalize_text(text: str, cfg: dict[str, Any]) -> str:
    """对单条文本做归一化。"""
    if cfg.get("strip_html", True):
        text = TAG_RE.sub(" ", text)
        text = html.unescape(text)

    if cfg.get("unicode_form"):
        text = unicodedata.normalize(cfg["unicode_form"], text)

    text = text.translate(ZERO_WIDTH)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if cfg.get("collapse_whitespace", True):
        text = WS_RE.sub(" ", text)
        text = MULTI_NL_RE.sub("\n\n", text)
        text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()


def normalize(
    samples: list[Sample], cfg: dict[str, Any], dag: DatasetDAG
) -> list[Sample]:
    inputs = sorted({s.head for s in samples if s.head})
    node_id = "xf:normalize"
    dag.add_node(
        LineageNode(
            node_id=node_id,
            kind="transform",
            label="Normalize",
            inputs=inputs,
            params=dict(cfg),
        )
    )

    kept: list[Sample] = []
    empty_dropped = 0
    seen_ids: dict[str, int] = {}

    for s in samples:
        s.text = normalize_text(s.text, cfg)
        if cfg.get("drop_empty", True) and not s.text:
            empty_dropped += 1
            continue

        # sample_id 指向「这一条记录」，content_digest 指向「这份内容」。
        # 内容完全相同的两条记录会共享 content_digest，但 sample_id 通过
        # 后缀 -2 / -3 保持唯一 —— 否则重复样本会在保留/丢弃两侧撞 id。
        base = f"{s.source_id}:{short_hash(s.text, 12)}"
        seen_ids[base] = seen_ids.get(base, 0) + 1
        s.sample_id = base if seen_ids[base] == 1 else f"{base}-{seen_ids[base]}"
        s.meta["content_digest"] = sha256_text(s.text)
        s.lineage.append(node_id)
        kept.append(s)

    dag.nodes[node_id].stats = {
        "in": len(samples),
        "out": len(kept),
        "empty_dropped": empty_dropped,
    }
    return kept
