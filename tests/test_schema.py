"""核心对象与血缘图的单元测试。"""

from __future__ import annotations

import pytest

from datasetfactory.lineage import DatasetDAG, LineageNode
from datasetfactory.schema import Dataset, Metric, Sample, Version, sha256_text, short_hash


# ----------------------------------------------------------------------
# schema
# ----------------------------------------------------------------------
def test_sha256_is_stable_and_content_sensitive():
    assert sha256_text("abc") == sha256_text("abc")
    assert sha256_text("abc") != sha256_text("abd")
    assert len(sha256_text("abc")) == 64
    assert len(short_hash("abc", 12)) == 12


def test_sample_metric_roundtrip():
    s = Sample(sample_id="a:1", text="hello", source_id="a")
    assert s.metric("missing") == 0.0
    assert s.metric("missing", default=-1.0) == -1.0
    assert not s.has_metric("quality.score")

    s.set_metric("quality.score", 0.75, chars=5)
    assert s.has_metric("quality.score")
    assert s.metric("quality.score") == pytest.approx(0.75)
    assert s.metrics["quality.score"].detail["chars"] == 5


def test_sample_head_follows_lineage():
    s = Sample(sample_id="a:1", text="x", source_id="a")
    assert s.head == ""
    s.lineage.extend(["src:a", "xf:normalize"])
    assert s.head == "xf:normalize"


def test_dataset_find_and_size():
    s1 = Sample(sample_id="a:1", text="x", source_id="a")
    s2 = Sample(sample_id="b:1", text="y", source_id="b")
    ver = Version(version_id="v1", parent=None, created_at="now", counts={"train": 1, "test": 1})
    ds = Dataset(dataset_id="d", version=ver, splits={"train": [s1], "test": [s2]})

    assert ds.size() == 2
    assert len(ds.all_samples()) == 2
    assert ds.find("a:1")[0] == "train"
    assert ds.find("b:1")[1] is s2
    assert ds.find("nope") is None


def test_metric_asdict_is_json_ready():
    import json

    m = Metric(name="q", value=0.5, detail={"a": 1})
    assert json.loads(json.dumps(m.to_dict()))["value"] == 0.5


# ----------------------------------------------------------------------
# lineage
# ----------------------------------------------------------------------
def test_dag_requires_known_parents():
    dag = DatasetDAG()
    dag.add_node(LineageNode("src:a", "source", "A"))
    with pytest.raises(ValueError, match="unknown parent"):
        dag.add_node(LineageNode("xf:1", "transform", "T", inputs=["src:missing"]))


def test_dag_rejects_duplicate_node():
    dag = DatasetDAG()
    dag.add_node(LineageNode("src:a", "source", "A"))
    with pytest.raises(ValueError, match="already exists"):
        dag.add_node(LineageNode("src:a", "source", "A"))


def test_dag_edges_and_traversal():
    dag = DatasetDAG()
    dag.add_node(LineageNode("src:a", "source", "A"))
    dag.add_node(LineageNode("src:b", "source", "B"))
    dag.add_node(LineageNode("xf:n", "transform", "N", inputs=["src:a", "src:b"]))
    dag.add_node(LineageNode("spl:train", "split", "Split/train", inputs=["xf:n"]))

    assert dag.parents("xf:n") == ["src:a", "src:b"]
    assert dag.children("src:a") == ["xf:n"]
    assert dag.ancestors("spl:train") == ["src:a", "src:b", "xf:n"]
    assert dag.descendants("src:a") == ["spl:train", "xf:n"]
    assert dag.sources() == ["src:a", "src:b"]
    assert dag.leaves() == ["spl:train"]


def test_dag_topological_order_respects_dependencies():
    dag = DatasetDAG()
    dag.add_node(LineageNode("src:a", "source", "A"))
    dag.add_node(LineageNode("xf:n", "transform", "N", inputs=["src:a"]))
    dag.add_node(LineageNode("ver:1", "version", "V", inputs=["xf:n"]))

    order = dag.topological()
    assert order.index("src:a") < order.index("xf:n") < order.index("ver:1")
    assert len(order) == 3


def test_dag_summary_and_exports():
    dag = DatasetDAG()
    dag.add_node(LineageNode("src:a", "source", "A", stats={"samples": 3}))
    dag.add_node(LineageNode("xf:n", "transform", "N", inputs=["src:a"], stats={"in": 3, "out": 2}))

    summary = dag.summary()
    assert summary["nodes"] == 2
    assert summary["edges"] == 1
    assert summary["by_kind"] == {"source": 1, "transform": 1}

    ascii_tree = dag.to_ascii()
    assert "Lineage DAG" in ascii_tree
    assert "src:a" in ascii_tree
    # 只有单输入，不该出现汇聚点小节
    assert "汇聚点" not in ascii_tree

    dot = dag.to_dot()
    assert dot.startswith("digraph lineage {")
    assert '"src:a" -> "xf:n"' in dot


def test_dag_ascii_lists_merge_points():
    dag = DatasetDAG()
    dag.add_node(LineageNode("src:a", "source", "A"))
    dag.add_node(LineageNode("src:b", "source", "B"))
    dag.add_node(LineageNode("xf:n", "transform", "N", inputs=["src:a", "src:b"]))

    assert "汇聚点" in dag.to_ascii()
    assert "src:a, src:b" in dag.to_ascii()
