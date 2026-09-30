"""数据谱系 DAG。

README「五、Lineage DAG」：

    Wikipedia-v1 ──┐
                   ├→ Clean-v1 → Dedup-v1 ─┐
    GitHub-v3 ─────┘                        ├→ Train-v4
    Books-v2 ─→ Clean-v2 ──────────────────┘

本模块只负责「图」本身——节点、边、祖先 / 后代查询、导出。
「回答三个问题」（Where from / What happened / Why here）的逻辑在 ``report.py``。

节点只增不改（append-only），边由 ``LineageNode.inputs`` 自动建立，
因此同一个 DAG 可以完整记录一次运行的全部血缘。
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .schema import utcnow


def _dot_escape(text: str) -> str:
    """转义 Graphviz DOT 双引号字符串里的特殊字符。

    反斜杠必须最先替换，否则会把后面替换出的引号反斜杠再转义一遍。
    换行转成 DOT 的 ``\\n`` 转义（标签内换行），避免把 label 拆成多行
    导致语法错误。
    """
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "")
    )


@dataclass
class LineageNode:
    """DAG 节点，串联 Source → Transform → Sample。

    kind 取值：``source`` | ``transform`` | ``filter`` | ``split`` | ``version``
    """

    node_id: str
    kind: str
    label: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetDAG:
    """不可变追加式血缘图。"""

    def __init__(self) -> None:
        self.nodes: dict[str, LineageNode] = {}
        self.edges: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    def add_node(self, node: LineageNode) -> str:
        if node.node_id in self.nodes:
            raise ValueError(f"node already exists: {node.node_id}")
        for parent in node.inputs:
            if parent not in self.nodes:
                raise ValueError(f"unknown parent '{parent}' for node '{node.node_id}'")
        self.nodes[node.node_id] = node
        for parent in node.inputs:
            self.nodes[parent].outputs.append(node.node_id)
            self.edges.append((parent, node.node_id))
        return node.node_id

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def parents(self, node_id: str) -> list[str]:
        return list(self.nodes[node_id].inputs)

    def children(self, node_id: str) -> list[str]:
        return list(self.nodes[node_id].outputs)

    def ancestors(self, node_id: str) -> list[str]:
        """所有上游节点（不含自身）。"""
        seen: set[str] = set()
        stack = list(self.parents(node_id))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self.parents(cur))
        return sorted(seen)

    def descendants(self, node_id: str) -> list[str]:
        """所有下游节点（不含自身）。"""
        seen: set[str] = set()
        stack = list(self.children(node_id))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self.children(cur))
        return sorted(seen)

    def topological(self) -> list[str]:
        """拓扑序；存在环则抛错。

        用 deque 而非 list：``list.pop(0)`` 每次要搬动整个列表（O(n)），
        节点多时整体退化成 O(n²)。deque.popleft() 是 O(1)，且同为 FIFO，
        出队顺序与原先逐字节一致（初始按 id 排序、子节点按 id 入队），
        因此拓扑序仍是确定的、可复现的。
        """
        indeg = {n: 0 for n in self.nodes}
        for _, dst in self.edges:
            indeg[dst] += 1
        queue = deque(sorted(n for n, d in indeg.items() if d == 0))
        order: list[str] = []
        while queue:
            cur = queue.popleft()
            order.append(cur)
            for child in sorted(self.children(cur)):
                indeg[child] -= 1
                if indeg[child] == 0:
                    queue.append(child)
        if len(order) != len(self.nodes):
            raise ValueError("cycle detected in lineage DAG")
        return order

    def by_kind(self, kind: str) -> list[str]:
        return sorted(n for n, node in self.nodes.items() if node.kind == kind)

    def sources(self) -> list[str]:
        return self.by_kind("source")

    def leaves(self) -> list[str]:
        return sorted(n for n, node in self.nodes.items() if not node.outputs)

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for node in self.nodes.values():
            kinds[node.kind] = kinds.get(node.kind, 0) + 1
        return {"nodes": len(self.nodes), "edges": len(self.edges), "by_kind": kinds}

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [self.nodes[n].to_dict() for n in self.topological()],
            "edges": [{"from": a, "to": b} for a, b in self.edges],
            "summary": self.summary(),
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def to_dot(self) -> str:
        """Graphviz DOT，便于渲染成图。

        node_id / label 里可能带引号或反斜杠（source_id、filter_id 都来自
        配置，是用户可控输入），直接插进双引号会生成非法 DOT、graphviz
        报语法错误。所有插值都过一遍 :func:`_dot_escape`。
        """
        lines = [
            "digraph lineage {",
            "  rankdir=LR;",
            '  node [shape=box, fontname="Helvetica"];',
        ]
        for node in self.nodes.values():
            # 先转义 label，再拼 DOT 的 \n —— 这个 \n 是换行转义，不能一起被转义掉
            label = f"{_dot_escape(node.label)}\\n({_dot_escape(node.kind)})"
            lines.append(f'  "{_dot_escape(node.node_id)}" [label="{label}"];')
        for a, b in self.edges:
            lines.append(f'  "{_dot_escape(a)}" -> "{_dot_escape(b)}";')
        lines.append("}")
        return "\n".join(lines)

    def to_ascii(self) -> str:
        """按拓扑序渲染，并单独列出「汇聚点」（多输入节点）。

        DAG 不是树，用缩进树渲染会丢失汇聚结构（多个来源合并到同一 transform），
        因此这里用「拓扑序清单 + 汇聚点清单」的表示法。
        """
        order = self.topological()
        if not order:
            return "(empty DAG)"

        id_w = max(len(n) for n in order)
        label_w = max(len(self.nodes[n].label) for n in order)
        kind_w = max(len(self.nodes[n].kind) for n in order)

        lines: list[str] = ["Lineage DAG（拓扑序）", ""]
        for nid in order:
            node = self.nodes[nid]
            stats = "  ".join(f"{k}={v}" for k, v in (node.stats or {}).items())
            lines.append(
                f"  {nid:<{id_w}}  {node.label:<{label_w}}  "
                f"[{node.kind:<{kind_w}}]  {stats}"
            )

        merges = [n for n in order if len(self.nodes[n].inputs) > 1]
        if merges:
            lines.append("")
            lines.append("汇聚点（多输入节点）：")
            for nid in merges:
                node = self.nodes[nid]
                lines.append(f"  {nid}  ← {', '.join(node.inputs)}")

        return "\n".join(lines)
