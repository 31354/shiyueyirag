# shiyueyirag

一个基于 **向量数据库（Milvus）+ RAG** 的智能问答系统。将文档切分、向量化后存入 Milvus，用户提问时检索相关片段，交给大模型生成回答。

> 支持 Agent 自主决策：根据问题类型选择 RAG 检索、Web 搜索或直接回答。

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Docker](https://img.shields.io/badge/docker-compose-blue)

## 功能特性

- 📄 **文档入库**：解析 PDF / DOCX / Markdown，切分后向量化存入 Milvus
- 🔍 **语义检索**：基于向量相似度召回相关文档片段
- 🤖 **RAG 问答**：检索结果 + 大模型生成，回答有据可依
- 🧠 **Agent 自主决策**：根据检索分数决定走 RAG、Web 搜索或直接回答
- 🌐 **API 服务**：提供 HTTP 接口，方便接入前端或其他系统
- 💬 **对话页面**：内置 `chat.html` 可直接测试
- 📊 **效果评估**：内置评估脚本，量化检索与问答质量
- 🐳 **容器化部署**：Docker Compose 一键拉起 Milvus / etcd / MinIO

## 架构图

```
                        ┌─────────────────┐
                        │   用户提问       │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Agent 决策循环  │
                        │  (tools.py)     │
                        └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
      ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
      │  RAG 检索     │   │  Web 搜索     │   │  直接回答     │
      │  (rag.py)    │   │  (tools.py)  │   │              │
      └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
             │                  │                  │
             ▼                  │                  │
      ┌──────────────┐          │                  │
      │   Milvus     │          │                  │
      │  向量数据库   │          │                  │
      └──────┬───────┘          │                  │
             │                  │                  │
             └──────────────────┼──────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  大模型生成答案   │
                       │  (智谱 API)     │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  带来源的回答     │
                       └─────────────────┘
```

**文档入库流程**：

```
文档 (PDF/DOCX/MD)
      │
      ▼
  解析 (jiexi.py)
      │
      ▼
  切分 (CHUNK_SIZE)
      │
      ▼
  向量化 (Embedding)
      │
      ▼
  Milvus 存储
```

## 工作流程

1. **文档入库**：解析文档 → 切分 → 向量化 → 存入 Milvus
2. **用户提问**：问题经 Embedding 转成向量
3. **语义检索**：在 Milvus 中召回 Top-K 相关片段
4. **Agent 决策**：根据相似度分数判断
   - 分数 ≥ 阈值 → 用 RAG 结果生成
   - 分数 < 阈值 → 尝试 Web 搜索或直接回答
5. **生成回答**：检索片段 + 问题交给大模型，返回带来源的答案

## 项目结构

```
shiyueyirag/
├── main.py              # 程序入口
├── api.py               # API 服务
├── rag.py               # RAG 检索逻辑
├── tools.py             # Agent 工具函数
├── jiexi.py             # 文档解析
├── wendang.py           # 文档处理
├── logger.py            # 日志
├── metrics.py           # 指标统计
├── eval.py              # 评估脚本
├── eval_agent.py        # Agent 评估
├── demo.py              # 示例
├── Dockerfile
├── docker-compose.yml   # Milvus / etcd / MinIO 编排
├── requirements.txt
├── .env.example         # 环境变量模板
└── chat.html            # 前端对话页面
```

## 环境要求

- Python 3.10+
- Docker & Docker Compose（用于 Milvus）
- 智谱 API Key（或其他兼容 OpenAI 协议的 LLM 服务）

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/31354/shiyueyirag.git
cd shiyueyirag
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填入：

```bash
ZHIPU_API_KEY=your_key_here
```

完整配置项见下方 [Configuration](#configuration)。

### 4. 启动 Milvus 向量数据库

```bash
docker compose up -d
```

启动后可访问：

- Milvus：`localhost:19530`
- MinIO 控制台：`localhost:9001`
- Attu（可选）：`localhost:8001`

### 5. 运行

```bash
# 命令行运行
python main.py

# 或启动 API 服务
python api.py
```

API 启动后访问 `http://localhost:8000/docs` 查看接口文档。

### 6. 打开对话页面

浏览器打开 `chat.html`，或通过 API 服务访问前端页面。

## Configuration

复制 `.env.example` 为 `.env`，按需修改。

### 必填项

| 变量 | 说明 |
|---|---|
| `ZHIPU_API_KEY` | 智谱 API Key，[获取地址](https://open.bigmodel.cn/) |

### LLM 配置

| 变量 | 说明 | 默认值 |
|---|---|---|
| `LLM_MODEL` | 生成答案使用的模型 | `glm-4-flash` |
| `TEMPERATURE` | 生成温度，越高越随机 | `0.2` |
| `MAX_TOKENS` | 单次生成最大 token 数 | `2048` |

### Embedding 配置

| 变量 | 说明 | 默认值 |
|---|---|---|
| `EMBEDDING_MODEL` | 向量化模型 | `embedding-2` |
| `EMBEDDING_DIM` | 向量维度，需与 Milvus 集合一致 | `1024` |

### Milvus 配置

| 变量 | 说明 | 默认值 |
|---|---|---|
| `MILVUS_HOST` | Milvus 服务地址 | `localhost` |
| `MILVUS_PORT` | Milvus 端口 | `19530` |
| `MILVUS_COLLECTION` | 集合名称 | `shiyue_docs` |

### RAG 检索配置

| 变量 | 说明 | 默认值 |
|---|---|---|
| `TOP_K` | 检索返回的文档条数 | `5` |
| `SCORE_THRESHOLD` | 相似度阈值，低于此值触发 Web 搜索 | `0.7` |
| `CHUNK_SIZE` | 文档分块大小（字符数） | `512` |
| `CHUNK_OVERLAP` | 分块重叠字符数 | `64` |

**关键参数说明**：

- **`SCORE_THRESHOLD`**：Agent 决策的核心参数。检索结果低于此阈值时，Agent 会尝试 Web 搜索或直接回答。调高更保守，调低更激进。
- **`TOP_K`**：影响召回率与延迟。调大召回更全但更慢，建议 3–5。
- **`CHUNK_SIZE`**：太小语义不完整，太大检索不精准。中文文档建议 400–600。

### Agent 配置

| 变量 | 说明 | 默认值 |
|---|---|---|
| `MAX_STEPS` | Agent 最大循环步数，防止死循环 | `5` |
| `ENABLE_WEB_SEARCH` | 是否启用 Web 搜索工具 | `false` |
| `WEB_SEARCH_API_KEY` | Web 搜索 API Key（启用时必填） | 无 |

### API 服务配置

| 变量 | 说明 | 默认值 |
|---|---|---|
| `API_HOST` | 监听地址 | `0.0.0.0` |
| `API_PORT` | 监听端口 | `8000` |

### 可观测性配置

| 变量 | 说明 | 默认值 |
|---|---|---|
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `LOG_DIR` | 日志输出目录 | `./logs` |
| `ENABLE_METRICS` | 是否开启指标统计 | `true` |

### 配置示例

**最小可运行配置**：

```bash
ZHIPU_API_KEY=your_key_here
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

**启用 Web 搜索**：

```bash
ZHIPU_API_KEY=your_key_here
ENABLE_WEB_SEARCH=true
WEB_SEARCH_API_KEY=your_search_key
SCORE_THRESHOLD=0.75
```

## Usage

### 命令行问答

```bash
python main.py
```

### API 调用

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "公司的报销流程是什么？"}'
```

返回示例：

```json
{
  "answer": "根据文档，报销流程分为三步……",
  "sources": [
    {"file": "制度手册.pdf", "score": 0.89},
    {"file": "财务指南.md", "score": 0.82}
  ],
  "steps": ["rag_search", "generate"]
}
```

## Evaluation

项目内置两套评估脚本，分别评估检索质量与 Agent 决策质量。

### 检索质量评估

```bash
python eval.py
```

输出指标：

| 指标 | 说明 | 目标值 |
|---|---|---|
| Hit Rate@K | Top-K 中是否包含正确答案 | ≥ 0.85 |
| MRR | 正确答案的平均排名倒数 | ≥ 0.70 |
| Context Precision | 检索片段的精确度 | ≥ 0.75 |
| Context Recall | 检索片段的召回率 | ≥ 0.80 |

### Agent 决策评估

```bash
python eval_agent.py
```

输出指标：

| 指标 | 说明 | 目标值 |
|---|---|---|
| 决策准确率 | Agent 是否选对工具（RAG/Web/直接） | ≥ 0.80 |
| 弃答率 | 不可回答问题是否正确拒答 | ≥ 0.90 |
| 平均步数 | Agent 完成任务的平均循环次数 | ≤ 3 |
| 幻觉率 | 生成内容无来源支撑的比例 | ≤ 0.10 |

> 目标值为参考基准，实际数值请以本地评测结果为准。

### 评测数据集

评测数据放在 `eval/datasets/qa.json`，格式：

```json
[
  {
    "question": "公司的报销流程是什么？",
    "answer": "报销流程分为三步……",
    "relevant_docs": ["制度手册.pdf"]
  }
]
```

### 评测结果

运行后结果输出到 `eval/results/`，包含：

- `rag_metrics.json`：检索指标
- `agent_metrics.json`：Agent 决策指标
- `report.md`：可读报告



## FAQ

### Q1: `eval.py` 的 Recall@1 全是 0，是检索坏了吗？

先别急着怀疑检索。最常见的原因是**评测集里的 `true_indices` 标错了**。

`eval.py` 用 `true_indices` 指定“正确答案在第几个 chunk”，如果标注的 id 和实际 chunk 内容对不上，检索再准也会被判为 0。

排查方法：

```bash
python -c "
import rag
rag.init()
for i, c in enumerate(rag._chunks):
    text = c if isinstance(c, str) else c.get('text', str(c))
    print(f'--- chunk[{i}] ---')
    print(text[:200])
"
```

对照 `eval_set` 里的 `true_indices`，确认每个问题对应的 chunk 内容是否真的相关。改对后 Recall 会立刻改善。

### Q2: 为什么 Rerank 后 Recall@3 反而下降了？

小样本评测（如 4 条 query）波动大，Rerank 模型对中文支持有限时可能把正确答案排后。建议：

- 扩大测试集到 20 条以上
- 换中文 reranker，如 `BAAI/bge-reranker-v2-m3`
- 调 `alpha` 权重，让 BM25 与向量检索的配比更合适

### Q3: 改了文档，为什么检索结果没变？

`eval.py` 启动时会检查 Milvus 中是否已有数据，如果已有块会**跳过 embedding**，导致新文档没被入库。

解决方法：清空 Milvus 数据卷后重新入库。

```bash
docker compose down
docker volume ls | Select-String milvus
docker volume rm <milvus数据卷名>
docker compose up -d
python eval.py
```

### Q4: 评测集只有 4 条 query，结果可信吗？

不太可信。4 条样本量太小，指标波动大，`Recall@1` 只能在 0、0.25、0.5、0.75、1.0 几档之间跳。

建议扩充到 20–30 条，覆盖不同类型问题（定义、对比、选型、误区等），指标才有参考价值。

### Q5: `Recall@1` 对多答案问题为什么算不准？

当一个问题对应多个 chunk（如 `FAISS 和 Milvus 有什么区别？` 对应 `chunk[3]` 和 `chunk[5]`），Top-1 只能命中一个，按命中比例算，R@1 最多 0.5。

这是评测算法的固有局限，不是检索问题。看 `Recall@3` 或 `Recall@5` 更合理。

### Q6: 启动时提示 `Failed to find CUDA` 怎么办？

这是警告，不是错误。如果没有 NVIDIA GPU 或没装 CUDA Toolkit，Rerank 模型会在 CPU 上运行，速度慢但不影响功能。

想消除警告：

- 安装 CUDA Toolkit 并设置 `CUDA_HOME` 环境变量
- 或忽略它，评测仍能正常跑完



## 技术栈

| 层级 | 技术 |
|---|---|
| 向量数据库 | Milvus + etcd + MinIO |
| LLM | 智谱 GLM |
| Embedding | 智谱 embedding-2 |
| API 框架 | FastAPI |
| 容器化 | Docker Compose |
| 前端 | 原生 HTML + JS |

## Roadmap

- [ ] 支持混合检索（向量 + BM25）
- [ ] 支持重排序（Rerank）
- [ ] 增加更多文档格式（Excel、PPT）
- [ ] 支持多轮对话
- [ ] 接入 Langfuse 做可观测性

## License

MIT
