"""Stage 3 — Fingerprint：SimHash（默认 64bit）+ 字符 shingle。

SimHash 的用途是近似重复检测：内容相近的文本，指纹的汉明距离很小。
相比 MinHash，SimHash 只需一个整数就能表示整条样本，便于写进 dataset.jsonl
并作为后续「内容寻址」的一部分。
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from ..lineage import DatasetDAG, LineageNode
from ..schema import Sample


def shingles(text: str, size: int = 5) -> list[str]:
    """按字符切 n-gram（先去掉所有空白，避免换行影响指纹）。"""
    compact = "".join(text.split())
    if not compact:
        return []
    if len(compact) <= size:
        return [compact]
    return [compact[i : i + size] for i in range(len(compact) - size + 1)]


def simhash(tokens: Iterable[str], bits: int = 64) -> int:
    """标准 SimHash：逐 token 投票，按位取符号。"""
    vector = [0] * bits
    n = 0
    for tok in tokens:
        h = int.from_bytes(
            hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for i in range(bits):
            vector[i] += 1 if (h >> i) & 1 else -1
        n += 1
    if n == 0:
        return 0
    out = 0
    for i in range(bits):
        if vector[i] > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    """两个指纹的汉明距离。

    用 ``int.bit_count()``（Python 3.10+，pyproject 的 requires-python 已保证）
    而不是 ``bin(x).count("1")``：前者是 C 层的 popcount，后者要为每个
    指纹分配一个临时字符串。near 去重要对每对样本算一次，样本多时这个
    差别会被放大。两者结果完全等价。
    """
    return (a ^ b).bit_count()


def fingerprint(
    samples: list[Sample], cfg: dict[str, Any], dag: DatasetDAG
) -> list[Sample]:
    algo = cfg.get("algo", "simhash")
    if algo != "simhash":
        raise ValueError(f"unsupported fingerprint algo: {algo}")

    bits = int(cfg.get("bits", 64))
    size = int(cfg.get("shingle_size", 5))

    inputs = sorted({s.head for s in samples if s.head})
    node_id = "xf:fingerprint"
    dag.add_node(
        LineageNode(
            node_id=node_id,
            kind="transform",
            label="Fingerprint",
            inputs=inputs,
            params={"algo": algo, "bits": bits, "shingle_size": size},
        )
    )

    for s in samples:
        toks = shingles(s.text, size)
        s.fingerprint = simhash(toks, bits)
        s.set_metric("fingerprint.shingles", float(len(toks)), bits=bits, size=size)
        s.lineage.append(node_id)

    dag.nodes[node_id].stats = {
        "in": len(samples),
        "out": len(samples),
        "bits": bits,
    }
    return samples
