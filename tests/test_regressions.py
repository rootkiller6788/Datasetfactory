"""针对已修缺陷的回归测试。

每条都对应一次具体修复，断言的是「坏输入不再产生坏行为」，
而不是「实现长什么样」—— 换实现时这些测试应当继续有效。
"""

from __future__ import annotations

import io
import json
import re
import sys

import pytest

from datasetfactory.cli import _force_utf8_stdio, main
from datasetfactory.config import load_config
from datasetfactory.lineage import DatasetDAG, LineageNode
from datasetfactory.pipeline import run_pipeline
from datasetfactory.report import explain_sample, render_explanation
from datasetfactory.stages.fingerprint import hamming
from datasetfactory.stages.ingest import load_jsonl


# ----------------------------------------------------------------------
# hamming：bit_count 与朴素数位实现等价
# ----------------------------------------------------------------------
def test_hamming_matches_naive_popcount():
    cases = [
        (0, 0),
        (0, (1 << 64) - 1),
        (1 << 63, (1 << 63) - 1),
        (0b1011, 0b0110),
    ]
    for a, b in cases:
        assert hamming(a, b) == bin(a ^ b).count("1"), (a, b)


def test_hamming_bounds():
    assert hamming(0, 0) == 0
    assert hamming(0, (1 << 64) - 1) == 64


# ----------------------------------------------------------------------
# to_dot：引号 / 反斜杠 / 换行必须转义
# ----------------------------------------------------------------------
def test_to_dot_escaping_is_structural():
    """含特殊字符时，每个 label 仍是一个合法的 DOT 双引号字符串。"""
    dag = DatasetDAG()
    dag.add_node(LineageNode('src:a"b', "source", 'Source(a"b)'))
    dag.add_node(
        LineageNode("xf:n", "transform", "C:\\path\nline2", inputs=['src:a"b'])
    )
    dot = dag.to_dot()

    # 按 DOT 的双引号字符串文法解析：内容是「非引号非反斜杠字符」或
    # 「反斜杠 + 任意字符」的序列。若有未转义的引号，它会让匹配提前
    # 收尾，剩下的那个引号就没法被任何匹配吞掉，下面的断言即失败。
    quoted = re.findall(r'"(?:[^"\\]|\\.)*"', dot)
    assert len(quoted) == 7, quoted
    assert '"' not in re.sub(r'"(?:[^"\\]|\\.)*"', "", dot)

    # 转义后的形态：\" 而不是裸 "；\\ 而不是裸 \
    assert 'label="Source(a\\"b)\\n(source)"' in dot
    assert 'label="C:\\\\path\\nline2\\n(transform)"' in dot
    # 字面换行会把 DOT 语句拆成两行，必须已经被转成 \n
    assert not any(
        line.strip().startswith("line2") for line in dot.splitlines()
    )


def test_to_dot_plain_labels_are_byte_identical():
    """无特殊字符时输出不应有任何变化。"""
    dag = DatasetDAG()
    dag.add_node(LineageNode("src:a", "source", "Source(a)"))
    dag.add_node(LineageNode("xf:n", "transform", "Normalize", inputs=["src:a"]))
    assert dag.to_dot() == (
        "digraph lineage {\n"
        "  rankdir=LR;\n"
        '  node [shape=box, fontname="Helvetica"];\n'
        '  "src:a" [label="Source(a)\\n(source)"];\n'
        '  "xf:n" [label="Normalize\\n(transform)"];\n'
        '  "src:a" -> "xf:n";\n'
        "}"
    )


# ----------------------------------------------------------------------
# 拓扑排序：deque 替换 pop(0) 后，顺序与覆盖性不变
# ----------------------------------------------------------------------
def test_topological_covers_every_node_once(result):
    order = result.dag.topological()
    assert len(order) == len(set(order))
    assert sorted(order) == sorted(result.dag.nodes)


def test_topological_respects_every_edge(result):
    """每个节点都必须排在其所有父节点之后。"""
    pos = {n: i for i, n in enumerate(result.dag.topological())}
    for parent, child in result.dag.edges:
        assert pos[parent] < pos[child], f"{parent} 应排在 {child} 之前"


def test_topological_is_deterministic(result):
    assert result.dag.topological() == result.dag.topological()


def test_topological_detects_cycle():
    dag = DatasetDAG()
    dag.add_node(LineageNode("a", "transform", "A"))
    dag.add_node(LineageNode("b", "transform", "B", inputs=["a"]))
    # 手工加一条 b -> a 的回边
    dag.nodes["a"].inputs.append("b")
    dag.nodes["b"].outputs.append("a")
    dag.edges.append(("b", "a"))
    with pytest.raises(ValueError, match="cycle"):
        dag.topological()


# ----------------------------------------------------------------------
# Transform id 必须能在 DAG 上找到对应节点
# ----------------------------------------------------------------------
def test_every_transform_id_resolves_to_a_dag_node(result):
    missing = [
        t.transform_id
        for t in result.dataset.transforms
        if t.transform_id not in result.dag.nodes
    ]
    assert not missing, f"这些 Transform 在 DAG 上没有对应节点: {missing}"


def test_filter_transform_id_follows_filter_id(cfg):
    """Filter 的 transform_id 不能被硬编码，必须跟着 filter_id 走。"""
    expected = f"flt:{cfg['filter']['filter_id']}"
    ids = {t.transform_id for t in run_pipeline(cfg).dataset.transforms}
    assert expected in ids


# ----------------------------------------------------------------------
# 配置校验：非法 filter op 在加载期就报错
# ----------------------------------------------------------------------
def test_invalid_filter_op_rejected_at_config_load(project_root, tmp_path):
    raw = json.loads(
        (project_root / "configs" / "default.json").read_text(encoding="utf-8")
    )
    raw["filter"]["rules"] = [{"metric": "quality.score", "op": "=>", "value": 0.5}]

    bad = tmp_path / "configs" / "bad.json"
    bad.parent.mkdir(parents=True)
    bad.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="op 不支持"):
        load_config(bad)


def test_valid_filter_ops_still_accepted(project_root, tmp_path):
    raw = json.loads(
        (project_root / "configs" / "default.json").read_text(encoding="utf-8")
    )
    raw["filter"]["rules"] = [
        {"metric": "quality.score", "op": ">", "value": 0.1},
        {"metric": "quality.score", "op": "<=", "value": 1.0},
        {"metric": "quality.score", "op": "!=", "value": 0.5},
    ]
    good = tmp_path / "configs" / "good.json"
    good.parent.mkdir(parents=True)
    good.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    assert load_config(good)["filter"]["rules"][0]["op"] == ">"


# ----------------------------------------------------------------------
# ingest：坏 jsonl 要报出文件与行号，且不能被当成对象
# ----------------------------------------------------------------------
def test_jsonl_syntax_error_reports_file_and_lineno(tmp_path):
    p = tmp_path / "broken.jsonl"
    p.write_text('{"text": "ok"}\n{"text": "bad",}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"broken\.jsonl:2"):
        load_jsonl(p, "text")


def test_jsonl_non_object_line_is_rejected(tmp_path):
    """裸字符串行原本会让 obj.get() 抛 AttributeError 穿透到用户面前。"""
    p = tmp_path / "scalar.jsonl"
    p.write_text('"just a string"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="期望 JSON 对象"):
        load_jsonl(p, "text")


def test_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "ok.jsonl"
    p.write_text('{"text": "a"}\n\n{"text": "b"}\n', encoding="utf-8")
    assert [t for t, _ in load_jsonl(p, "text")] == ["a", "b"]


# ----------------------------------------------------------------------
# 控制台编码：GBK 码表装不下 ✓ / ✗
# ----------------------------------------------------------------------
def test_gbk_stream_cannot_encode_check_mark():
    """记录这个缺陷的根因，防止有人把 _force_utf8_stdio 当成多余代码删掉。"""
    with pytest.raises(UnicodeEncodeError):
        io.TextIOWrapper(io.BytesIO(), encoding="gbk").write("✓")


def test_force_utf8_stdio_reconfigures_stdout(monkeypatch):
    buf = io.BytesIO()
    fake_out = io.TextIOWrapper(buf, encoding="gbk")
    fake_err = io.TextIOWrapper(io.BytesIO(), encoding="gbk")
    monkeypatch.setattr(sys, "stdout", fake_out)
    monkeypatch.setattr(sys, "stderr", fake_err)

    _force_utf8_stdio()

    assert fake_out.encoding.lower().replace("-", "") == "utf8"
    print("✓ ✗")  # 修复前这里会抛 UnicodeEncodeError
    fake_out.flush()
    assert "✓".encode("utf-8") in buf.getvalue()


def test_force_utf8_stdio_tolerates_streams_without_reconfigure(monkeypatch):
    """pytest 的捕获对象与 StringIO 没有 reconfigure，不能因此报错。"""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    _force_utf8_stdio()  # 不应抛异常


def test_render_explanation_marks_rules_with_check_symbols(result):
    sid = result.dataset.all_samples()[0].sample_id
    text = render_explanation(explain_sample(result, sid))
    assert "✓" in text
    assert "Why is it here?" in text


def test_cli_explain_kept_sample_exits_zero(result, project_root, capsys):
    sid = result.dataset.all_samples()[0].sample_id
    rc = main(
        [
            "explain",
            "--config",
            str(project_root / "configs" / "default.json"),
            "--sample-id",
            sid,
        ]
    )
    assert rc == 0
    assert sid in capsys.readouterr().out


def test_cli_explain_unknown_sample_exits_nonzero(project_root, capsys):
    rc = main(
        [
            "explain",
            "--config",
            str(project_root / "configs" / "default.json"),
            "--sample-id",
            "no:such:id",
        ]
    )
    assert rc == 1


# ----------------------------------------------------------------------
# 包入口
# ----------------------------------------------------------------------
def test_module_entry_point_exists():
    import datasetfactory.__main__ as entry

    assert callable(entry.main)
