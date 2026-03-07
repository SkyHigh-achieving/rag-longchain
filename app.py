import os

# --- Local Environment Isolation ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "vector_db")

for d in [MODELS_DIR, DATA_DIR, VECTOR_DB_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

os.environ["HF_HOME"] = MODELS_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODELS_DIR
os.environ["TRANSFORMERS_CACHE"] = MODELS_DIR

from dotenv import load_dotenv
load_dotenv()

import torch
import gradio as gr
from core.processor import DocumentProcessor
from core.retriever import AdvancedRetriever
from core.generator import Generator


# Initialize components
processor = DocumentProcessor()
retriever = None
generator = Generator(
    model_name=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

def is_model_ready(model_dir):
    return os.path.isfile(os.path.join(model_dir, "model.safetensors")) or os.path.isfile(os.path.join(model_dir, "pytorch_model.bin"))

def process_pdfs(file_list):
    """Handles PDF uploads and initializes the retriever."""
    global retriever
    if not file_list:
        return "请先上传 PDF 文件。"
    
    all_docs = []
    for file in file_list:
        # file is a temp file object from Gradio
        docs = processor.load_pdf(file.name)
        all_docs.extend(docs)
    
    # Initialize the Advanced Retriever (Hybrid Search + Reranker)
    # Highlight: This is much better than simple vector search!
    retriever = AdvancedRetriever(
        all_docs,
        embedding_model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        reranker_model_name=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    )
    
    rerank_status = "已启用" if retriever.reranker_available else "未启用（将使用 Hybrid Top-K）"
    return f"成功加载 {len(file_list)} 个文件，共 {len(all_docs)} 个文本切分片段。重排模型：{rerank_status}"

def query_system(query, top_k=5, mode="Advanced"):
    """Handles user queries and returns answers with context."""
    if not retriever:
        return "请先上传并解析 PDF。", ""
    if not query or not str(query).strip():
        return "请输入问题后再提问。", ""
    
    # Advanced Retrieval with Rerank
    if mode == "Advanced":
        context_docs = retriever.retrieve_with_rerank(query, top_k=top_k)
    else:
        # Simulate standard RAG (Vector only)
        context_docs = retriever.vector_retriever.invoke(query)[:top_k]
    
    context_display = "\n\n---\n\n".join([f"**Source {i+1}**: {doc.page_content}" for i, doc in enumerate(context_docs)])
    
    try:
        answer = generator.generate(query, context_docs)
    except Exception as e:
        err_text = str(e)
        if "Connection error" in err_text:
            fallback = context_docs[0].page_content[:500] if context_docs else "未检索到可用片段。"
            answer = (
                "LLM 服务当前不可连接（通常是本地 Ollama 未启动或 API 地址不可达）。\n\n"
                "先给出基于检索结果的参考摘要：\n"
                f"{fallback}"
            )
        else:
            answer = f"生成阶段失败：{err_text}"
    
    return answer, context_display

# Gradio Interface
with gr.Blocks(title="高级学术 RAG 系统 (浙大/清华面试版)") as demo:
    gr.Markdown("# 🎓 高级学术论文 RAG 知识库系统")
    gr.Markdown("""
    **面试亮点**:
    1. **Hybrid Search**: 混合检索 (BM25 + Vector) 捕捉学术专有名词。
    2. **BGE-Reranker**: 交叉熵重排序，比单纯余弦相似度更精准。
    3. **Academic Prompting**: 严谨的学术风格提示词，杜绝幻觉。
    """)
    
    # Display System Status
    llm_source = "本地模型 (Local Ollama/Llama)" if generator.is_local else "云端模型 (OpenAI API)"
    hw_status = "🚀 GPU (CUDA) 加速" if torch.cuda.is_available() else "💻 CPU 模式 (已优化多线程)"
    
    # Check Model Readiness
    emb_ready = is_model_ready(os.path.join(MODELS_DIR, "bge-small-en-v1.5"))
    rerank_ready = is_model_ready(os.path.join(MODELS_DIR, "bge-reranker-base"))
    model_status = "✅ 模型已就绪" if emb_ready and rerank_ready else "⚠️ 模型加载中/缺失"
    
    gr.Markdown(f"**当前系统状态**: 🟢 {llm_source} | 🟢 {hw_status} | 🟢 {model_status}")

    
    with gr.Tab("1. 文件上传"):
        file_input = gr.File(label="上传学术 PDF", file_count="multiple")
        upload_btn = gr.Button("解析并构建知识库")
        status_out = gr.Textbox(label="状态")
        upload_btn.click(process_pdfs, inputs=[file_input], outputs=[status_out])
        
    with gr.Tab("2. 智能问答"):
        with gr.Row():
            with gr.Column(scale=2):
                query_in = gr.Textbox(label="请输入您的问题", placeholder="例如：这篇文章的核心创新点是什么？")
                mode_choice = gr.Radio(["Advanced", "Standard"], label="检索模式 (Advanced 开启重排序)", value="Advanced")
                k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="检索片段数量")
                ask_btn = gr.Button("提问", variant="primary")
                answer_out = gr.Markdown(label="回答内容")
            with gr.Column(scale=1):
                context_out = gr.Markdown(label="检索参考内容")
        
        ask_btn.click(query_system, inputs=[query_in, k_slider, mode_choice], outputs=[answer_out, context_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000)
