# MIMIC-IV PrimeKG 子图提取工具

本项目用于从 PrimeKG 知识图谱中提取与 MIMIC-IV 患者相关的动态子图，支持临床预测任务（诊断预测、用药推荐）。

## 项目概述

将 MIMIC-IV 电子健康记录（EHR）中的医学文本实体映射到 PrimeKG 知识图谱，并为每个预测任务提取相关的子图。关键特性：

- **动态时间切片**：只使用 `context_end` 之前的数据，避免数据泄露
- **UMLS 映射**：通过 UMLS 将临床文本映射到 PrimeKG 实体
- **N-hop 子图提取**：提取种子节点周围的 N 跳邻域
- **并行处理**：支持多进程加速

## 核心脚本

### 1. `build_mapping.py`
构建 UMLS 文本到 PrimeKG 实体的映射字典。

**功能**：
- 从 UMLS 数据库加载医学概念
- 映射诊断（disease/phenotype）和药物（drug）实体
- 支持 PAR 层级向上查找父疾病节点
- 输出：`outputs/umls_to_primekg.pkl`

**运行**：
```bash
# 设置环境变量
export MONGO_URI="mongodb://user:pass@host:port/db?params"
export MONGO_DB_NAME="umls_test"

# 运行映射构建
python build_mapping.py
```

**输出统计**：
- 诊断词条：~40,000 个，映射成功率 ~85%
- 药物词条：~20,000 个，映射成功率 ~75%
- 输出文件：`/data/MIMIC/primekg_subgraphs/umls_to_primekg.pkl`

### 2. `extract_subgraphs.py`
为每个预测任务提取 PrimeKG 子图。

**功能**：
- 加载患者的 UMLS 概念序列（动态时间切片）
- 映射到 PrimeKG 实体索引
- 提取 N-hop 子图（只保留有意义的关系类型）
- 保存为 JSON 格式

**运行**：
```bash
# 诊断预测任务
python extract_subgraphs.py --task diagnoses_icd --hops 2 --workers 8

# 用药推荐任务
python extract_subgraphs.py --task prescriptions_atc --hops 2 --workers 8

# 测试模式（限制任务数）
python extract_subgraphs.py --task diagnoses_icd --hops 2 --max_tasks 1000
```

**参数说明**：
- `--task`：任务类型（`diagnoses_icd` 或 `prescriptions_atc`）
- `--hops`：子图跳数（默认 2）
- `--workers`：并行进程数（默认 1）
- `--max_tasks`：限制处理的任务数（用于测试）

### 3. `viewer_app.py`
交互式 Web 可视化应用（基于 Streamlit）。

**功能**：
- 浏览所有任务的子图文件
- 交互式 3D 网络可视化
- 查看节点和边的详细信息
- 分析种子节点之间的路径
- 导出子图数据

**运行**：
```bash
# 启动可视化应用
streamlit run viewer_app.py

# 指定端口
streamlit run viewer_app.py --server.port 8501
```

**功能特性**：
- **文件浏览器**：按任务类型、患者 ID、住院 ID 筛选
- **3D 可视化**：种子节点高亮显示，支持交互式探索
- **路径分析**：查看种子节点之间的连接路径
- **统计信息**：节点类型分布、关系类型分布
- **数据导出**：下载节点和边的详细信息

**输出格式**：
```json
{
  "task_id": "10000883_25221576_18",
  "subject_id": 10000883,
  "hadm_id": 25221576,
  "context_end": 18,
  "seed_nodes": ["14024", "28956"],
  "nodes": [
    {
      "id": "14024",
      "name": "Lisinopril",
      "type": "drug",
      "source": "DrugBank"
    }
  ],
  "edges": [
    {
      "source": "14024",
      "target": "28956",
      "relation": "indication"
    }
  ]
}
```

## 数据路径

### 输入数据
```
/data/ehr/MIMIC/data_processed/
├── task_index/
│   ├── diagnoses_icd.csv          # 325,675 个任务
│   └── prescriptions_atc.csv      # 1,579,422 个任务
└── concept_sequences/event_concepts/
    └── {subject_id}.pkl           # 患者的 UMLS 概念序列

/data/literature_and_kg/primeKg/
├── primekg_graph.gpickle          # PrimeKG 图（129K 节点，8M 边）
└── PrimeKg.csv                    # PrimeKG 原始数据
```

### 输出数据
```
/data/MIMIC/primekg_subgraphs/
├── umls_to_primekg.pkl            # UMLS 映射字典（59,682 个实体）
├── primekg_node_attrs.pkl         # PrimeKG 节点属性缓存
├── diagnoses_icd/                 # 325,675 个 JSON 文件
│   └── {subject_id}_{hadm_id}_{context_end}.json
└── prescriptions_atc/             # 1,579,422 个 JSON 文件
    └── {subject_id}_{hadm_id}_{context_end}.json
```

## 技术细节

### 保留的关系类型
为避免子图过于密集，只保留以下有意义的关系：
- `indication`：药物适应症
- `contraindication`：药物禁忌症
- `disease_phenotype_positive/negative`：疾病-表型关联
- `disease_protein`：疾病-蛋白质关联
- `drug_protein`：药物-蛋白质关联
- `drug_effect`：药物效果
- `disease_disease`：疾病-疾病关联
- `drug_drug`：药物协同作用
- `exposure_disease`：暴露-疾病关联

### 映射策略
- **诊断字段**：通过 UMLS CUI → MONDO/HPO → PrimeKG disease/phenotype
  - 找不到时沿 PAR 层级向上查找父疾病（最多 5 层）
- **药物字段**：通过 UMLS CUI → DrugBank → PrimeKG drug
  - 找不到时直接跳过

## 依赖环境

```bash
# Python 3.10+
pip install networkx pandas pymongo tqdm

# 可视化应用额外依赖
pip install streamlit plotly
```

## 性能指标

- **映射构建**：约 10-15 分钟（一次性）
- **输出大小**：
  - diagnoses_icd：~1.4 GB
  - prescriptions_atc：~6.4 GB

## 参考文档

详细的技术报告请参考：[REPORT_PrimeKG_Subgraph_Extraction.md](REPORT_PrimeKG_Subgraph_Extraction.md)

## 注意事项

1. **数据隐私**：确保 MIMIC-IV 数据使用符合 PhysioNet 协议
2. **MongoDB 配置**：运行 `build_mapping.py` 前需配置 UMLS 数据库连接
3. **磁盘空间**：确保 `/data/MIMIC/` 有足够空间（~10 GB）
4. **内存需求**：建议至少 16 GB RAM（加载 PrimeKG 图需要 ~8 GB）

## 许可证

本项目仅用于学术研究，请遵守相关数据使用协议。
