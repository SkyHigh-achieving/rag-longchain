import os
import re
import json
import math
import time
import csv
import io
import datetime
from urllib.parse import quote
from urllib.request import urlopen

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

def translate_to_chinese(text):
    text = (text or "").strip()
    if not text:
        return text
    blocks = []
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for line in lines:
        if len(line) <= 380:
            blocks.append(line)
        else:
            parts = re.split(r'(?<=[\.\!\?。！？])\s*', line)
            buf = ""
            for p in parts:
                if not p:
                    continue
                if len(buf) + len(p) <= 380:
                    buf += p
                else:
                    if buf:
                        blocks.append(buf)
                    buf = p
            if buf:
                blocks.append(buf)
    translated = []
    for block in blocks:
        try:
            api = f"https://api.mymemory.translated.net/get?q={quote(block)}&langpair=en|zh-CN"
            with urlopen(api, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                zh = data.get("responseData", {}).get("translatedText", "").strip()
            translated.append(zh if zh else block)
        except Exception:
            translated.append(block)
    return "\n".join(translated)

def build_retrieval_fallback(context_docs, output_lang):
    snippets = []
    for i, doc in enumerate(context_docs[:3]):
        clean = " ".join((doc.page_content or "").split())
        snippets.append(f"{i+1}. {clean[:260]}")
    body = "\n".join(snippets) if snippets else "未检索到可用片段。"
    if output_lang != "英文":
        body = translate_to_chinese(body)
        return "⚠️ **LLM 服务当前不可连接**\n\n原因：当前 `OPENAI_API_BASE / OPENAI_API_KEY / MODEL_NAME` 配置不可用。\n\n**解决建议：**\n1. 优先使用**云端 API**：在 `.env` 中配置可用的 `OPENAI_API_BASE`、`OPENAI_API_KEY` 与 `MODEL_NAME`。\n2. 如果使用**本地 OpenAI 兼容服务**：请先启动本地服务，并确认 `.env` 中地址正确。\n\n--- \n**基于检索结果的中文摘要：**\n" + body
    return "⚠️ **LLM is unavailable**\n\nCurrent `OPENAI_API_BASE / OPENAI_API_KEY / MODEL_NAME` config is unavailable.\n\n**Solution:**\n1. Prefer **cloud API**: set valid `OPENAI_API_BASE`, `OPENAI_API_KEY`, and `MODEL_NAME` in `.env`.\n2. If using a **local OpenAI-compatible service**, start it first and ensure the endpoint is correct.\n\n---\n**Evidence snippets:**\n" + body

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

def query_system(query, top_k=5, mode="Advanced", output_lang="中文"):
    if not retriever:
        yield "请先上传并解析 PDF。", ""
        return
    if not query or not str(query).strip():
        yield "请输入问题后再提问。", ""
        return

    if mode == "Advanced":
        context_docs = retriever.retrieve_with_rerank(query, top_k=top_k)
    else:
        context_docs = retriever.vector_retriever.invoke(query)[:top_k]

    context_display = "\n\n".join([f"[证据 {i+1}]\n{doc.page_content}" for i, doc in enumerate(context_docs)])
    yield "正在生成回答，请稍候...", context_display

    try:
        stream_enabled = hasattr(generator, "stream_generate")
        answer = ""
        if stream_enabled and output_lang == "英文":
            for chunk in generator.stream_generate(query, context_docs):
                if not chunk:
                    continue
                answer += chunk
                yield answer, context_display
            return
        if stream_enabled:
            answer = "".join(generator.stream_generate(query, context_docs))
        else:
            answer = generator.generate(query, context_docs)
        if output_lang != "英文":
            answer = translate_to_chinese(answer)
            yield answer, context_display
            return
        progressive = ""
        chunk_size = 12
        for i in range(0, len(answer), chunk_size):
            progressive += answer[i:i + chunk_size]
            yield progressive, context_display
    except Exception as e:
        err_text = str(e)
        lower_err = err_text.lower()
        if any(k in lower_err for k in ["connection", "connect", "timeout", "api key", "authentication", "401", "403", "404", "429"]):
            answer = build_retrieval_fallback(context_docs, output_lang)
        else:
            answer = f"生成阶段失败：{err_text}"
        yield answer, context_display

def inspect_attention(query, top_k=3, mode="Advanced"):
    if not retriever:
        return "请先上传并解析 PDF。"
    if not query or not str(query).strip():
        return "请输入问题后再分析。"
    if not getattr(generator, "use_local_hf", False):
        return "当前仅本地 HuggingFace 模式支持注意力分析。"
    if mode == "Advanced":
        context_docs = retriever.retrieve_with_rerank(query, top_k=top_k)
    else:
        context_docs = retriever.vector_retriever.invoke(query)[:top_k]
    return generator.analyze_attention(query, context_docs, top_n=min(3, int(top_k)))

def _normalize_for_overlap(text):
    text = (text or "").lower()
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if len(t) >= 2]

def _top1_hit(answer, top1_text):
    ans_tokens = set(_normalize_for_overlap(answer))
    doc_tokens = set(_normalize_for_overlap(top1_text))
    if not ans_tokens or not doc_tokens:
        return False, 0.0
    inter = len(ans_tokens & doc_tokens)
    ratio = inter / max(len(ans_tokens), 1)
    return ratio >= 0.12, ratio

def _measure_perf(query, context_docs):
    start = time.perf_counter()
    first_token_ts = None
    chunks = []
    for chunk in generator.stream_generate(query, context_docs):
        if first_token_ts is None and str(chunk).strip():
            first_token_ts = time.perf_counter()
        chunks.append(chunk)
    end = time.perf_counter()
    answer = "".join(chunks)
    ttft = (first_token_ts - start) if first_token_ts is not None else (end - start)
    gen_window = max(end - (first_token_ts or start), 1e-6)
    if getattr(generator, "use_local_hf", False):
        token_count = len(generator.local_tokenizer.encode(answer, add_special_tokens=False))
    else:
        token_count = max(len(answer.split()), 1)
    tps = token_count / gen_window
    return answer, float(ttft), float(tps)

def _svg_line_chart(title, x_vals, series_dict):
    width, height = 760, 280
    left, right, top, bottom = 56, 24, 36, 44
    inner_w = width - left - right
    inner_h = height - top - bottom
    if not x_vals:
        return "<div>无可用数据</div>"
    x_min, x_max = min(x_vals), max(x_vals)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    y_min, y_max = 0.0, 1.0
    def sx(x):
        return left + (x - x_min) * inner_w / (x_max - x_min)
    def sy(y):
        y = max(y_min, min(y_max, y))
        return top + (y_max - y) * inner_h / (y_max - y_min)
    colors = ["#2563eb", "#ef4444", "#16a34a", "#a855f7"]
    names = list(series_dict.keys())
    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    svg.append(f'<text x="{left}" y="22" font-size="14" fill="#111827">{title}</text>')
    svg.append(f'<line x1="{left}" y1="{top+inner_h}" x2="{left+inner_w}" y2="{top+inner_h}" stroke="#9ca3af"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+inner_h}" stroke="#9ca3af"/>')
    for i in range(6):
        yv = i / 5
        yy = sy(yv)
        svg.append(f'<line x1="{left}" y1="{yy}" x2="{left+inner_w}" y2="{yy}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="10" y="{yy+4}" font-size="11" fill="#6b7280">{yv:.1f}</text>')
    for x in x_vals:
        xx = sx(x)
        svg.append(f'<line x1="{xx}" y1="{top}" x2="{xx}" y2="{top+inner_h}" stroke="#f3f4f6"/>')
        svg.append(f'<text x="{xx-4}" y="{top+inner_h+18}" font-size="11" fill="#6b7280">{x}</text>')
    for i, name in enumerate(names):
        color = colors[i % len(colors)]
        points = " ".join([f"{sx(x_vals[j]):.2f},{sy(series_dict[name][j]):.2f}" for j in range(len(x_vals))])
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        legend_x = left + 8 + i * 180
        legend_y = height - 12
        svg.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+18}" y2="{legend_y}" stroke="{color}" stroke-width="2.2"/>')
        svg.append(f'<text x="{legend_x+24}" y="{legend_y+4}" font-size="11" fill="#374151">{name}</text>')
    svg.append("</svg>")
    return "".join(svg)

def compare_attention_modes(query, top_k=3):
    if not retriever:
        return "请先上传并解析 PDF。"
    if not query or not str(query).strip():
        return "请输入问题后再评估。"
    if not getattr(generator, "use_local_hf", False):
        return "当前仅本地 HuggingFace 模式支持该评估。"
    k = int(top_k)
    k_values = list(range(1, max(2, min(k, 6)) + 1))
    adv_conc, std_conc = [], []
    adv_hit_curve, std_hit_curve = [], []
    adv_tps_curve, std_tps_curve = [], []
    adv_ttft_curve, std_ttft_curve = [], []
    csv_rows = [["k", "strategy", "concentration", "entropy", "ttft", "tps", "hit_ratio"]]
    last_adv = None
    last_std = None
    last_adv_hit = (False, 0.0)
    last_std_hit = (False, 0.0)
    for kk in k_values:
        docs_adv = retriever.retrieve_with_rerank(query, top_k=kk)
        docs_std = retriever.vector_retriever.invoke(query)[:kk]
        adv = generator.analyze_attention_struct(query, docs_adv, top_n=min(3, kk))
        std = generator.analyze_attention_struct(query, docs_std, top_n=min(3, kk))
        if not adv.get("ok") or not std.get("ok"):
            return f"评估失败：Advanced={adv.get('message', '')} | Standard={std.get('message', '')}", "<div>注意力曲线生成失败</div>"
        answer_adv, adv_ttft, adv_tps = _measure_perf(query, docs_adv)
        answer_std, std_ttft, std_tps = _measure_perf(query, docs_std)
        adv_top1 = adv["ranked"][0] if adv["ranked"] else {"index": -1, "text": ""}
        std_top1 = std["ranked"][0] if std["ranked"] else {"index": -1, "text": ""}
        adv_hit, adv_ratio = _top1_hit(answer_adv, adv_top1["text"])
        std_hit, std_ratio = _top1_hit(answer_std, std_top1["text"])
        
        # Calculate entropies for CSV
        adv_ent = -sum(item["weight"] * math.log(max(item["weight"], 1e-9)) for item in adv["all_scores"])
        std_ent = -sum(item["weight"] * math.log(max(item["weight"], 1e-9)) for item in std["all_scores"])
        
        csv_rows.append([kk, "Advanced", adv["concentration"], adv_ent, adv_ttft, adv_tps, adv_ratio])
        csv_rows.append([kk, "Standard", std["concentration"], std_ent, std_ttft, std_tps, std_ratio])
        
        adv_conc.append(adv["concentration"])
        std_conc.append(std["concentration"])
        adv_hit_curve.append(min(1.0, adv_ratio * 2.0))
        std_hit_curve.append(min(1.0, std_ratio * 2.0))
        adv_tps_curve.append(adv_tps)
        std_tps_curve.append(std_tps)
        adv_ttft_curve.append(adv_ttft)
        std_ttft_curve.append(std_ttft)
        last_adv = adv
        last_std = std
        last_adv_hit = (adv_hit, adv_ratio)
        last_std_hit = (std_hit, std_ratio)
    max_tps = max(adv_tps_curve + std_tps_curve + [1e-6])
    max_ttft = max(adv_ttft_curve + std_ttft_curve + [1e-6])
    adv_tps_norm = [x / max_tps for x in adv_tps_curve]
    std_tps_norm = [x / max_tps for x in std_tps_curve]
    adv_ttft_norm = [1.0 - (x / max_ttft) for x in adv_ttft_curve]
    std_ttft_norm = [1.0 - (x / max_ttft) for x in std_ttft_curve]
    adv_entropy = -sum(item["weight"] * math.log(max(item["weight"], 1e-9)) for item in last_adv["all_scores"])
    std_entropy = -sum(item["weight"] * math.log(max(item["weight"], 1e-9)) for item in last_std["all_scores"])
    adv_top1 = last_adv["ranked"][0] if last_adv["ranked"] else {"index": -1}
    std_top1 = last_std["ranked"][0] if last_std["ranked"] else {"index": -1}
    lines = [
        "Advanced vs Standard 注意力与证据一致性对照：",
        f"- Advanced 注意力集中度: {last_adv['concentration']:.3f} | 熵: {adv_entropy:.3f}",
        f"- Standard 注意力集中度: {last_std['concentration']:.3f} | 熵: {std_entropy:.3f}",
        f"- Advanced Top1证据命中: {'是' if last_adv_hit[0] else '否'} | 重叠率: {last_adv_hit[1]:.3f} | 证据ID: {adv_top1['index']}",
        f"- Standard Top1证据命中: {'是' if last_std_hit[0] else '否'} | 重叠率: {last_std_hit[1]:.3f} | 证据ID: {std_top1['index']}",
        f"- Performance@k={k_values[-1]}: Advanced TTFT={adv_ttft_curve[-1]:.2f}s, TPS={adv_tps_curve[-1]:.2f} | Standard TTFT={std_ttft_curve[-1]:.2f}s, TPS={std_tps_curve[-1]:.2f}",
        "- 解释：集中度越高代表注意力更聚焦；Top1命中越高代表回答与最关注证据更一致。"
    ]
    chart_1 = _svg_line_chart(
        "解释性曲线（随检索片段数 k 变化）",
        k_values,
        {
            "Advanced-集中度": adv_conc,
            "Standard-集中度": std_conc,
            "Advanced-Top1命中代理": adv_hit_curve,
            "Standard-Top1命中代理": std_hit_curve
        }
    )
    chart_2 = _svg_line_chart(
        "性能曲线（归一化，越高越好）",
        k_values,
        {
            "Advanced-TPS": adv_tps_norm,
            "Standard-TPS": std_tps_norm,
            "Advanced-TTFT*": adv_ttft_norm,
            "Standard-TTFT*": std_ttft_norm
        }
    )
    
    csv_path = os.path.join(DATA_DIR, "research_results.csv")
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
    except Exception:
        csv_path = None

    # Generate HTML Report
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>RAG 模型注意力机制深度对照报告</title>
<style>
  body {{ font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; color: #333; }}
  h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
  .summary {{ background: #f9f9f9; padding: 20px; border-radius: 8px; border-left: 5px solid #007bff; white-space: pre-wrap; font-family: monospace; font-size: 14px; }}
  .charts {{ margin-top: 40px; text-align: center; }}
  .chart-container {{ margin-bottom: 40px; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
  svg {{ max-width: 100%; height: auto; }}
  .timestamp {{ color: #999; font-size: 12px; margin-top: 50px; text-align: right; border-top: 1px solid #eee; padding-top: 10px; }}
</style>
</head>
<body>
  <h1>RAG 模型注意力机制深度对照报告</h1>
  
  <h2>1. 研究结论摘要</h2>
  <div class="summary">{"".join(lines)}</div>
  
  <h2>2. 可视化分析图表</h2>
  <div class="charts">
    <div class="chart-container">
      <h3>解释性曲线（Cognitive Behavior）</h3>
      {chart_1}
    </div>
    <div class="chart-container">
      <h3>性能曲线（System Engineering）</h3>
      {chart_2}
    </div>
  </div>
  
  <div class="timestamp">
    生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
  </div>
</body>
</html>"""
    
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    html_filename = f"research_report_{timestamp_str}.html"
    html_path = os.path.join(DATA_DIR, html_filename)
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception:
        html_path = None

    return "\n".join(lines), chart_1 + chart_2, csv_path, html_path

# Gradio Interface
with gr.Blocks(title="论文问答系统") as demo:
    gr.Markdown("# 📚 论文问答系统")
    
    # Display System Status
    llm_source = "本地 HuggingFace 模型" if getattr(generator, "use_local_hf", False) else ("本地 OpenAI 兼容服务" if generator.is_local else "云端模型 (OpenAI 兼容 API)")
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
        gr.Markdown("检索策略说明：Advanced = BM25+向量+重排；Standard = 仅向量。检索片段数量是送入模型的证据条数，数量越大覆盖越广但速度越慢。")
        with gr.Row():
            with gr.Column(scale=2):
                query_in = gr.Textbox(label="请输入您的问题", placeholder="例如：这篇文章的核心创新点是什么？")
                mode_choice = gr.Radio(["Advanced", "Standard"], label="检索策略", value="Advanced")
                lang_choice = gr.Radio(["中文", "英文"], label="输出语言", value="中文")
                k_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="检索片段数量（证据条数）")
                ask_btn = gr.Button("提问", variant="primary")
                answer_out = gr.Textbox(label="回答内容", lines=14, max_lines=28)
                attn_btn = gr.Button("分析 Attention 焦点")
                attn_out = gr.Textbox(label="Attention 解释结果", lines=8, max_lines=14)
                compare_btn = gr.Button("对照评估 Advanced vs Standard")
                compare_out = gr.Textbox(label="研究对照结果", lines=8, max_lines=14)
                with gr.Row():
                    csv_file_out = gr.File(label="下载研究原始数据 (CSV)")
                    html_report_out = gr.File(label="下载完整研究报告 (HTML)")
                compare_curve_out = gr.HTML(label="研究对照曲线")
            with gr.Column(scale=1):
                context_out = gr.Textbox(label="检索证据片段（Source）", lines=18, max_lines=36)
        
        ask_btn.click(query_system, inputs=[query_in, k_slider, mode_choice, lang_choice], outputs=[answer_out, context_out])
        attn_btn.click(inspect_attention, inputs=[query_in, k_slider, mode_choice], outputs=[attn_out])
        compare_btn.click(compare_attention_modes, inputs=[query_in, k_slider], outputs=[compare_out, compare_curve_out, csv_file_out, html_report_out])

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=8000)
