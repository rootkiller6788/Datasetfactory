# DatasetFactory 数据集工厂

> 来源：factory-everything/factory-everything-项目总表.md ｜ 板块：factory-everything · L0 ｜ 落地约束：4060 8G 单机可跑

## 一、条目信息（原始字段，完整保留）

| 字段 | 内容 |
| --- | --- |
| 项目 | DatasetFactory 数据集工厂 |
| 类型 | Factory |
| 真正有意思的核心 | 去重、质量评分、污染检测、数据谱系 DAG |

---

## 二、核心命题

不是"数据清洗脚本"。真正值得做的是 **Lineage DAG**——让任何一个 sample 都能回答三个问题：

> Where did this sample come from? What happened to it? Why is it here?

## 三、整体架构

```
                 DatasetFactory
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
     Ingest         Transform       Metadata
        │              │              │
        ▼              ▼              ▼
  JSON/CSV/HF      Normalize      Source/License
  Web/Files        Clean          Hash/Version
        │              │              │
        └───────┬──────┴──────────────┘
                ▼
          Canonical Samples
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
      Dedup   Quality  Contamination
        │       │        │
        └───────┼────────┘
                ▼
            Filter Engine
                │
                ▼
            Dataset DAG
                │
                ▼
        Versioned Dataset
                │
                ▼
        Report / Export
```

## 四、主流水线

```
Raw Sources
    ↓
Ingest
    ↓
Normalize
    ↓
Fingerprint
    ↓
Deduplicate
    ↓
Quality Score
    ↓
Contamination Check
    ↓
Filter
    ↓
Split
    ↓
Version
    ↓
Dataset Artifact
```

## 五、Lineage DAG

```
Wikipedia-v1 ──┐
               ├→ Clean-v1 → Dedup-v1 ─┐
GitHub-v3 ─────┘                        ├→ Train-v4
Books-v2 ─→ Clean-v2 ──────────────────┘
```

- 每个节点是一次 Transform，每条边是"输入 → 输出"的数据流向
- 任一 sample 持有指向其来源节点的引用，可反向遍历出完整血缘链
- 这是区别于普通"数据清洗脚本"的关键：**产物可审计、可复现、可归因**

## 六、核心对象

| 对象 | 职责 |
| --- | --- |
| Dataset | 一次产物的容器，绑定版本与 split |
| Sample | 最小数据单元，携带 lineage 引用 |
| Source | 原始来源（路径 / 授权 / 抓取时间） |
| Transform | 一次可复现的处理步骤（含参数） |
| Metric | 质量 / 去重 / 污染的度量定义 |
| Filter | 基于 Metric 的保留/丢弃规则 |
| Split | train / valid / test 切分策略 |
| Version | 不可变的数据集快照 |
| LineageNode | DAG 节点，串联 Source → Transform → Sample |

## 七、阶段性落地计划

- **M0｜Canonical Schema 与 Ingest**：定义 Sample / Source / Transform 的 canonical schema，跑通 JSON / CSV / HF / Web / Files 五类输入的 Ingest（非自动）。
- **M1｜指纹与去重**：实现 Fingerprint（MinHash / SimHash）+ Exact / Fuzzy Dedup 主链路，输出去重前后统计。
- **M2｜质量与污染**：接入 Quality Score（规则 + 模型）与 Contamination Check（n-gram / 嵌入比对），Filter Engine 阈值参数化，由评分器驱动搜索。
- **M3｜Lineage DAG 与发布**：构建 Dataset DAG，任意 sample 可回溯来源与全部 Transform，产出 Versioned Dataset + Report / Export。

## 八、验收标准

能端到端完成「DatasetFactory 数据集工厂」，并在其「核心」（去重、质量评分、污染检测、数据谱系 DAG）上给出可复现的量化结果：

- 去重率、质量分分布、污染命中率可量化复现
- 任取一个 sample，可在 Lineage DAG 上完整回答"从哪来、经过了什么、为什么在这里"

## 附录 A、工程骨架

```
L0-datasetfactory/
├── README.md                 本设计文档
├── main.py                   零安装入口（无需 pip install）
├── pyproject.toml            打包与 pytest 配置
├── requirements.txt          依赖说明（核心链路零依赖）
├── .gitignore                忽略 __pycache__ / .pytest_cache / out 等
├── configs/
│   └── default.json          流水线配置（来源 / 阈值 / 权重 / 切分）
├── data/
│   ├── raw/                  示例原始数据
│   │   ├── wiki.jsonl        jsonl 来源（含 exact / near 重复各一对）
│   │   ├── books.csv         csv 来源（含一条低质量记录）
│   │   ├── papers.jsonl      jsonl 来源
│   │   └── notes/            dir 来源（含一条污染记录）
│   └── eval/
│       └── benchmark.jsonl   评测集，用于污染检测
├── src/datasetfactory/
│   ├── schema.py             核心对象：Source / Sample / Metric / Filter / Split / Version / Rejection
│   ├── lineage.py            LineageNode + DatasetDAG
│   ├── config.py             配置加载与校验
│   ├── pipeline.py           九段流水线编排
│   ├── report.py             报告生成 + 血缘三问 + 落盘
│   ├── cli.py                run / explain / dag / list
│   └── stages/               九个 stage，每个都是 (samples, cfg, dag) -> ...
│       ├── ingest.py         ① 读入 jsonl / csv / dir（hf / web 为扩展点）
│       ├── normalize.py      ② NFKC / 去 HTML / 压缩空白
│       ├── fingerprint.py    ③ SimHash 64bit + 字符 shingle
│       ├── dedup.py          ④ Exact(sha256) + Near(汉明距离)
│       ├── quality.py        ⑤ 四子分启发式打分
│       ├── contamination.py  ⑥ 与评测集做 n-gram 重合度
│       ├── filter.py         ⑦ 规则表达式过滤（AND 语义）
│       ├── split.py          ⑧ 哈希确定性切分
│       └── version.py        ⑨ 内容寻址版本
├── tests/                    pytest 单测 + 端到端测试
└── out/                      运行产物（每次 run 重建）
```

## 附录 B、快速开始

```bash
# 核心链路零第三方依赖，直接用标准库跑
python main.py run --config configs/default.json

# 回答某个 sample 的三个问题
python main.py explain --config configs/default.json --sample-id wiki:226c66118f44

# 查看 Lineage DAG（文本 / Graphviz DOT）
python main.py dag --config configs/default.json
python main.py dag --config configs/default.json --dot > out/lineage.dot

# 列出保留与丢弃的样本
python main.py list --config configs/default.json
```

跑完 `out/` 下会得到：

| 产物 | 内容 |
| --- | --- |
| `report.md` | 漏斗 / 来源 / 切分 / 质量分布 / 丢弃明细 / DAG |
| `dataset.jsonl` | 全量保留样本（含 split、metrics、filter_evidence、lineage） |
| `splits/*.jsonl` | 按 train / valid / test 分文件 |
| `rejections.jsonl` | 全部丢弃记录及原因 |
| `lineage.json` | 完整血缘图（节点 + 边） |
| `lineage.dot` | Graphviz 源文件 |
| `version.json` | 版本号与内容 digest |

运行测试：

```bash
pip install pytest
pytest
```

核心链路零第三方依赖；只有跑测试才需要 `pytest`。

## 附录 C、已验证状态

本骨架已实际跑通，不是纸面设计：

- `pytest` → **60 passed**，覆盖 schema / lineage / 九个 stage / 端到端流水线与可复现性
- `python main.py run` → 原始 17 条 → 最终 13 条，丢弃 4 条，切分 9 / 2 / 2，DAG 16 节点 15 边
- 4 条丢弃各自对应一类规则：`near` 判重、`exact` 判重、质量硬失败、污染超阈
- `explain` 三个问题均能作答；保留样本的链路以版本节点收尾
  （`src:papers → … → spl:test → ver:test:v1`）

**关于 near 去重阈值**：SimHash 汉明距离强烈依赖文本长度，不存在对长短文本都合适的
绝对阈值。实测（64bit + 字符 5-shingle）：187 shingle 的长文本「改 1 个词 ≈ 6、
改 3 个词 ≈ 13」，而无关文本落在 26~35；同一处改动在 41 shingle 的短文本上会产生
约 2 倍距离。默认取保守的 `8` —— **漏检只是少删一条，误合并会静默丢掉一条真实数据**。
以长文本为主时可放宽到 10~12；短文本占比高时应先加一道最小长度门槛，
跳过指纹本身不可靠的样本。详见 `src/datasetfactory/stages/dedup.py` 的模块文档。

---

_本 README 的「条目信息」字段来自 `factory-everything-项目总表.md`，架构与流水线为 L0 深化设计稿；可按实际进展修订。_
