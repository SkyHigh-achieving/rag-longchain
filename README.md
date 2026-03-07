<<<<<<< HEAD
# Advanced Academic RAG System

> **Graduate Interview Project** (Designed for THU/ZJU/PKU CS Recruitment)
> 
> 本项目是一款专为学术论文设计的、具备“检索-重排-生成”三层架构的高性能 RAG 系统。

## 🌟 核心亮点 (Core Highlights)

1. **高级混合检索 (Hybrid Retrieval)**:
   - 结合了 **BM25** (关键词匹配) 与 **FAISS** (向量语义搜索)。
   - 专门优化了学术专有名词的捕捉精度，比单纯向量搜索更适合处理严谨的学术文档。

2. **交叉熵重排序 (BGE-Reranker)**:
   - 集成 **BAAI/bge-reranker-base** 模型。
   - 对初步检索到的文档片段进行二次精排，有效过滤掉语义相关的非核心内容，极大地降低了大模型的幻觉概率。

3. **硬件自适应与环境隔离**:
   - 自动识别 CPU/GPU 设备，针对 CPU 多线程推理进行了底层优化。
   - 实现 **Full Local Isolation**：所有模型权重、依赖库、缓存数据均被重定向至工程目录内，确保了系统的可移植性。

4. **双模生成引擎**:
   - 支持 OpenAI 云端 API 与本地 Llama/Ollama 模型无缝切换。

## 🛠️ 技术架构 (Architecture)

- **Retrieval Pipeline**: BM25 + Vector -> Ensemble -> BGE-Reranker (Top-K)
- **Embedding**: `BAAI/bge-small-en-v1.5`
- **Reranker**: `BAAI/bge-reranker-base`
- **Frontend**: Gradio Academic Interface
- **Backend**: LangChain + Torch + FAISS

## 🚀 快速启动

1. **安装依赖**:
   ```powershell
   ./setup_local.ps1
   ```
2. **下载模型**:
   ```powershell
   .\.venv\Scripts\python.exe download_models.py
   ```
3. **运行演示**:
   ```powershell
   .\.venv\Scripts\python.exe app.py
   ```

## 📝 面试 Q&A 准备

- **Q: 为什么在有了向量搜索后还需要重排层 (Reranker)?**
  - *A: 向量搜索本质是计算余弦相似度，对语义相似但逻辑不相关的片段区分度较低。Reranker 通过交叉熵模型深度计算 Query 与 Doc 的相关性，能显著提升输入 LLM 的 Context 质量。*

- **Q: 系统如何处理大规模 PDF 的解析?**
  - *A: 项目使用了多级切分策略，在保持段落完整性的同时通过叠加 Overlap 机制避免了上下文截断导致的语义丢失。*
=======
# rag-longchain
本地化检索增强生成系统
>>>>>>> 92417278e49d5fb6d9e3947d2cc036acec82f28d
