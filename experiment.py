"""
experiment.py — Advanced RAG vs Standard RAG 消融实验脚本
===========================================================
用法:
    # 基础运行（使用 demo PDF + 默认问题集）
    python experiment.py

    # 自定义 PDF 和问题
    python experiment.py --pdf data/my_paper.pdf --questions questions.txt

    # 不上传 wandb（本地保存结果）
    python experiment.py --no-wandb

环境要求:
    pip install wandb  (已在 requirements.txt 中)
"""

import os
import sys
import math
import time
import json
import csv
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ─────────────────────────────────────────────
# 0. 路径隔离（与 app.py 保持一致）
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_DIR     = PROJECT_ROOT / "data"
VECTOR_DB_DIR= PROJECT_ROOT / "vector_db"
RESULTS_DIR  = PROJECT_ROOT / "experiment_results"

for d in [MODELS_DIR, DATA_DIR, VECTOR_DB_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"]                   = str(MODELS_DIR)
os.environ["SENTENCE_TRANSFORMERS_HOME"]= str(MODELS_DIR)
os.environ["TRANSFORMERS_CACHE"]        = str(MODELS_DIR)

# ─────────────────────────────────────────────
# 1. 日志
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / "experiment.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("experiment")

# ─────────────────────────────────────────────
# 2. 默认测试问题集（针对 demo_paper.pdf）
#    —— 如果你有自己的论文，在 --questions 文件里每行一个问题
# ─────────────────────────────────────────────
DEFAULT_QUESTIONS = [
    "What is the main contribution of this paper?",
    "How does the hybrid retrieval method work?",
    "What is the role of BM25 in this system?",
    "What are the experimental results showing?",
    "How does the reranker improve retrieval quality?",
    "What datasets were used for evaluation?",
    "What is the advantage of using a Cross-Encoder reranker?",
    "How does this approach handle technical terminology like LoRA or QLoRA?",
    "What are the limitations mentioned in the paper?",
    "What future work is proposed?",
]

# ─────────────────────────────────────────────
# 3. 辅助函数（直接复用 app.py 中经过验证的逻辑）
# ─────────────────────────────────────────────
def _normalize_tokens(text: str) -> List[str]:
    """简单词级别 normalize，用于命中率计算。"""
    import re
    text = (text or "").lower()
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if len(t) >= 2]


def _top1_hit_ratio(answer: str, top1_text: str) -> float:
    """answer 与 top1 证据文本的 token 重叠率（Jaccard-like）。"""
    ans_tokens = set(_normalize_tokens(answer))
    doc_tokens = set(_normalize_tokens(top1_text))
    if not ans_tokens or not doc_tokens:
        return 0.0
    inter = len(ans_tokens & doc_tokens)
    return inter / max(len(ans_tokens), 1)


def _calc_entropy(all_scores: List[Dict]) -> float:
    """注意力权重分布的香农熵（越低代表越集中）。"""
    return -sum(
        item["weight"] * math.log(max(item["weight"], 1e-9))
        for item in all_scores
    )


def _measure_generation(
    generator,
    query: str,
    context_docs: list,
) -> Tuple[str, float, float]:
    """
    返回 (answer, ttft_seconds, tps)
    ttft = Time To First Token
    tps  = Tokens Per Second（近似，按空格分词）
    """
    start = time.perf_counter()
    first_token_ts: Optional[float] = None
    chunks: List[str] = []

    for chunk in generator.stream_generate(query, context_docs):
        if first_token_ts is None and str(chunk).strip():
            first_token_ts = time.perf_counter()
        chunks.append(chunk)

    end = time.perf_counter()
    answer = "".join(chunks)

    ttft = (first_token_ts - start) if first_token_ts else (end - start)
    gen_window = max(end - (first_token_ts or start), 1e-6)

    # 如果有本地 tokenizer 则精确计算，否则按空格近似
    if getattr(generator, "use_local_hf", False) and hasattr(generator, "local_tokenizer"):
        token_count = len(generator.local_tokenizer.encode(answer, add_special_tokens=False))
    else:
        token_count = max(len(answer.split()), 1)

    tps = token_count / gen_window
    return answer, float(ttft), float(tps)


# ─────────────────────────────────────────────
# 4. 单个 Query 的完整实验单元
# ─────────────────────────────────────────────
def run_single_query(
    query: str,
    retriever,
    generator,
    k: int,
    strategy: str,        # "advanced" | "standard"
    use_attention: bool,
) -> Dict:
    """
    对一个 query + k + strategy 的组合运行实验，返回指标字典。
    """
    result = {
        "query": query,
        "k": k,
        "strategy": strategy,
        "concentration": None,
        "entropy": None,
        "top1_hit_ratio": None,
        "ttft": None,
        "tps": None,
        "answer_len": None,
        "error": None,
    }

    try:
        # ── 检索 ──
        if strategy == "advanced":
            context_docs = retriever.retrieve_with_rerank(query, top_k=k)
        else:
            context_docs = retriever.vector_retriever.invoke(query)[:k]

        if not context_docs:
            result["error"] = "no_docs"
            return result

        # ── 注意力分析（仅本地 HF 模式支持）──
        if use_attention and getattr(generator, "use_local_hf", False):
            attn = generator.analyze_attention_struct(
                query, context_docs, top_n=min(3, k)
            )
            if attn.get("ok"):
                result["concentration"] = attn["concentration"]
                result["entropy"]       = _calc_entropy(attn["all_scores"])
                top1 = attn["ranked"][0] if attn.get("ranked") else {"text": ""}
            else:
                top1 = {"text": ""}
        else:
            top1 = {"text": context_docs[0].page_content if context_docs else ""}

        # ── 生成 + 性能 ──
        answer, ttft, tps = _measure_generation(generator, query, context_docs)

        result["ttft"]          = ttft
        result["tps"]           = tps
        result["answer_len"]    = len(answer.split())
        result["top1_hit_ratio"]= _top1_hit_ratio(answer, top1.get("text", ""))

    except Exception as e:
        result["error"] = str(e)
        log.warning(f"  ✗ [{strategy}@k={k}] query='{query[:40]}' error: {e}")

    return result


# ─────────────────────────────────────────────
# 5. 主实验循环
# ─────────────────────────────────────────────
def run_experiment(
    pdf_path: str,
    questions: List[str],
    k_range: range,
    use_wandb: bool,
    wandb_project: str,
    use_attention: bool,
    embedding_model: str,
    reranker_model: str,
):
    # ── 初始化 wandb ──
    if use_wandb:
        try:
            import wandb
            run = wandb.init(
                project=wandb_project,
                name=f"ablation-RAG-{time.strftime('%m%d-%H%M')}",
                config={
                    "k_range":        f"{k_range.start}-{k_range.stop-1}",
                    "num_questions":  len(questions),
                    "embedding_model":embedding_model,
                    "reranker_model": reranker_model,
                    "strategies":     ["advanced", "standard"],
                },
            )
            log.info(f"✅ WandB 已连接: {run.url}")
        except ImportError:
            log.warning("wandb 未安装，跳过上传。pip install wandb")
            use_wandb = False
        except Exception as e:
            log.warning(f"wandb.init 失败: {e}，改为本地模式。")
            use_wandb = False

    # ── 加载组件 ──
    log.info("正在加载模型（首次运行会下载，请耐心等待）...")

    from dotenv import load_dotenv
    load_dotenv()

    from core.processor import DocumentProcessor
    from core.retriever  import AdvancedRetriever
    from core.generator  import Generator

    processor = DocumentProcessor()
    docs = processor.load_pdf(pdf_path)
    log.info(f"PDF 已加载: {len(docs)} 个文本片段（来自 {pdf_path}）")

    retriever = AdvancedRetriever(
        docs,
        embedding_model_name=embedding_model,
        reranker_model_name=reranker_model,
    )
    rerank_ok = getattr(retriever, "reranker_available", False)
    log.info(f"检索器就绪 | Reranker: {'✅' if rerank_ok else '⚠️ 未加载，Advanced 将退化为 Hybrid-only'}")

    generator = Generator(
        model_name=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    use_local_hf = getattr(generator, "use_local_hf", False)
    log.info(f"生成器就绪 | 模式: {'本地HF' if use_local_hf else '云端API'}")

    if use_attention and not use_local_hf:
        log.warning("⚠️ Attention 分析仅支持本地 HF 模式，当前为 API 模式，自动跳过注意力指标。")
        use_attention = False

    # ── 实验主循环 ──
    all_results: List[Dict] = []
    strategies = ["advanced", "standard"]

    total = len(questions) * len(k_range) * len(strategies)
    done  = 0

    log.info(f"\n{'='*60}")
    log.info(f"开始实验: {len(questions)} 问题 × k={k_range.start}..{k_range.stop-1} × {len(strategies)} 策略 = {total} 次运行")
    log.info(f"{'='*60}\n")

    for qi, query in enumerate(questions):
        log.info(f"[问题 {qi+1}/{len(questions)}] {query[:60]}")

        for k in k_range:
            for strategy in strategies:
                log.info(f"  → strategy={strategy:8s}  k={k}")

                result = run_single_query(
                    query=query,
                    retriever=retriever,
                    generator=generator,
                    k=k,
                    strategy=strategy,
                    use_attention=use_attention,
                )
                all_results.append(result)
                done += 1

                # ── WandB 逐步记录 ──
                if use_wandb and not result.get("error"):
                    prefix = f"{strategy}/k{k}"
                    log_dict = {
                        f"{prefix}/ttft":           result["ttft"],
                        f"{prefix}/tps":            result["tps"],
                        f"{prefix}/answer_len":     result["answer_len"],
                        f"{prefix}/top1_hit_ratio": result["top1_hit_ratio"],
                    }
                    if result["concentration"] is not None:
                        log_dict[f"{prefix}/concentration"] = result["concentration"]
                        log_dict[f"{prefix}/entropy"]       = result["entropy"]

                    # 对比指标（同 k，不同策略）—— 每轮 standard 记录后对比
                    if strategy == "standard":
                        # 找到同 query、同 k 的 advanced 结果
                        adv = next(
                            (r for r in all_results
                             if r["query"] == query and r["k"] == k
                             and r["strategy"] == "advanced"
                             and not r.get("error")),
                            None,
                        )
                        if adv:
                            log_dict[f"compare/k{k}/ttft_delta"] = (
                                (result["ttft"] or 0) - (adv["ttft"] or 0)
                            )
                            log_dict[f"compare/k{k}/tps_delta"] = (
                                (result["tps"] or 0) - (adv["tps"] or 0)
                            )
                            log_dict[f"compare/k{k}/hit_ratio_delta"] = (
                                (result["top1_hit_ratio"] or 0)
                                - (adv["top1_hit_ratio"] or 0)
                            )

                    import wandb as _wandb
                    _wandb.log(log_dict, step=done)

                # 进度
                pct = done / total * 100
                log.info(
                    f"  ✓ ttft={result.get('ttft', 'N/A'):.2f}s  "
                    f"tps={result.get('tps', 'N/A'):.1f}  "
                    f"hit={result.get('top1_hit_ratio', 'N/A'):.3f}  "
                    f"[{pct:.0f}%]"
                    if not result.get("error") else
                    f"  ✗ error={result['error']}"
                )

    # ── 聚合统计（按策略 × k 分组，取平均）──
    log.info("\n" + "="*60)
    log.info("实验完成，正在生成聚合统计...")

    agg: Dict[str, Dict[int, Dict[str, List]]] = {
        s: {k: {"ttft":[], "tps":[], "hit":[], "conc":[], "ent":[]}
            for k in k_range}
        for s in strategies
    }
    for r in all_results:
        if r.get("error"):
            continue
        s, k = r["strategy"], r["k"]
        agg[s][k]["ttft"].append(r["ttft"])
        agg[s][k]["tps"].append(r["tps"])
        agg[s][k]["hit"].append(r["top1_hit_ratio"])
        if r["concentration"] is not None:
            agg[s][k]["conc"].append(r["concentration"])
            agg[s][k]["ent"].append(r["entropy"])

    def mean(lst): return sum(lst) / len(lst) if lst else None

    # 打印对比表
    header = f"{'k':>3} | {'Adv TTFT':>9} {'Adv TPS':>8} {'Adv Hit':>8} | {'Std TTFT':>9} {'Std TPS':>8} {'Std Hit':>8}"
    log.info(header)
    log.info("-" * len(header))

    agg_rows = []
    for k in k_range:
        a, s_ = agg["advanced"][k], agg["standard"][k]
        row = {
            "k": k,
            "adv_ttft": mean(a["ttft"]),  "adv_tps": mean(a["tps"]),  "adv_hit": mean(a["hit"]),
            "std_ttft": mean(s_["ttft"]), "std_tps": mean(s_["tps"]), "std_hit": mean(s_["hit"]),
            "adv_conc": mean(a["conc"]),  "adv_ent": mean(a["ent"]),
            "std_conc": mean(s_["conc"]), "std_ent": mean(s_["ent"]),
        }
        agg_rows.append(row)

        def fmt(v): return f"{v:.3f}" if v is not None else "  N/A "
        log.info(
            f"{k:>3} | {fmt(row['adv_ttft']):>9} {fmt(row['adv_tps']):>8} {fmt(row['adv_hit']):>8}"
            f" | {fmt(row['std_ttft']):>9} {fmt(row['std_tps']):>8} {fmt(row['std_hit']):>8}"
        )

        # WandB 聚合指标（均值曲线，横轴是 k）
        if use_wandb:
            import wandb as _wandb
            agg_log = {"k": k}
            for prefix, data in [("advanced_avg", row["adv_ttft"]), ("standard_avg", row["std_ttft"])]:
                pass
            _wandb.log({
                "agg/k":                  k,
                "agg/advanced/ttft_mean": row["adv_ttft"],
                "agg/advanced/tps_mean":  row["adv_tps"],
                "agg/advanced/hit_mean":  row["adv_hit"],
                "agg/advanced/conc_mean": row["adv_conc"],
                "agg/advanced/ent_mean":  row["adv_ent"],
                "agg/standard/ttft_mean": row["std_ttft"],
                "agg/standard/tps_mean":  row["std_tps"],
                "agg/standard/hit_mean":  row["std_hit"],
                "agg/standard/conc_mean": row["std_conc"],
                "agg/standard/ent_mean":  row["std_ent"],
                # 关键对比：Advanced 的提升量
                "agg/delta/hit_ratio":    (
                    (row["adv_hit"] or 0) - (row["std_hit"] or 0)
                    if row["adv_hit"] is not None and row["std_hit"] is not None else None
                ),
            })

    # ── 保存结果 CSV ──
    ts = time.strftime("%Y%m%d_%H%M%S")
    raw_csv = RESULTS_DIR / f"raw_results_{ts}.csv"
    agg_csv = RESULTS_DIR / f"agg_results_{ts}.csv"

    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query","k","strategy","concentration","entropy",
            "top1_hit_ratio","ttft","tps","answer_len","error"
        ])
        writer.writeheader()
        writer.writerows(all_results)

    with open(agg_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)

    log.info(f"\n原始数据已保存: {raw_csv}")
    log.info(f"聚合数据已保存: {agg_csv}")

    # WandB 上传 CSV artifact
    if use_wandb:
        import wandb as _wandb
        artifact = _wandb.Artifact("ablation_results", type="dataset")
        artifact.add_file(str(raw_csv))
        artifact.add_file(str(agg_csv))
        _wandb.log_artifact(artifact)
        _wandb.finish()
        log.info("✅ 结果已上传至 WandB，请在浏览器查看曲线。")

    # ── 最终摘要 ──
    log.info("\n" + "="*60)
    log.info("📊 关键发现摘要（面试可直接引用）：")
    for row in agg_rows:
        k = row["k"]
        if row["adv_hit"] is not None and row["std_hit"] is not None:
            delta_hit = (row["adv_hit"] - row["std_hit"]) * 100
            sign = "↑" if delta_hit >= 0 else "↓"
            log.info(
                f"  k={k}: Advanced Top1命中率 {sign}{abs(delta_hit):.1f}% "
                f"vs Standard  |  "
                f"Adv TPS {row['adv_tps']:.1f} vs Std TPS {row['std_tps']:.1f}"
            )
    log.info("="*60)

    return agg_rows


# ─────────────────────────────────────────────
# 6. CLI 入口
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="RAG 消融实验脚本")
    p.add_argument("--pdf",      default="data/demo_paper.pdf",
                   help="PDF 路径（默认使用 demo_paper.pdf）")
    p.add_argument("--questions",default=None,
                   help="问题文件路径，每行一个问题（默认使用内置10题）")
    p.add_argument("--k-min",    type=int, default=1,  help="检索 k 最小值")
    p.add_argument("--k-max",    type=int, default=6,  help="检索 k 最大值（含）")
    p.add_argument("--no-wandb", action="store_true",  help="禁用 WandB 上传")
    p.add_argument("--no-attention", action="store_true", help="禁用 Attention 分析（节省时间）")
    p.add_argument("--wandb-project", default="RAG-Ablation-Study",
                   help="WandB 项目名")
    p.add_argument("--embedding-model", default=None,
                   help="覆盖 .env 中的 EMBEDDING_MODEL")
    p.add_argument("--reranker-model",  default=None,
                   help="覆盖 .env 中的 RERANKER_MODEL")
    return p.parse_args()


def main():
    args = parse_args()

    # 加载问题集
    if args.questions:
        with open(args.questions, encoding="utf-8") as f:
            questions = [ln.strip() for ln in f if ln.strip()]
    else:
        questions = DEFAULT_QUESTIONS

    # 模型名从 .env 或参数读取
    from dotenv import load_dotenv
    load_dotenv()
    embedding_model = args.embedding_model or os.getenv(
        "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
    )
    reranker_model = args.reranker_model or os.getenv(
        "RERANKER_MODEL", "BAAI/bge-reranker-base"
    )

    # 检查 PDF
    pdf_path = args.pdf
    if not os.path.exists(pdf_path):
        log.warning(f"PDF 不存在: {pdf_path}，正在生成 demo PDF...")
        try:
            import subprocess
            subprocess.run(
                [sys.executable, "create_dummy_pdf.py"],
                check=True
            )
            log.info("demo PDF 生成完毕。")
        except Exception as e:
            log.error(f"无法生成 demo PDF: {e}")
            sys.exit(1)

    log.info(f"""
╔══════════════════════════════════════════════════════╗
║   Advanced RAG vs Standard RAG — 消融实验            ║
╠══════════════════════════════════════════════════════╣
║  PDF:       {pdf_path:<42}║
║  问题数:    {len(questions):<42}║
║  k 范围:    {args.k_min} ~ {args.k_max:<40}║
║  WandB:     {'开启' if not args.no_wandb else '关闭':<42}║
║  Attention: {'开启' if not args.no_attention else '关闭':<42}║
║  Embedding: {embedding_model:<42}║
║  Reranker:  {reranker_model:<42}║
╚══════════════════════════════════════════════════════╝
""")

    run_experiment(
        pdf_path        = pdf_path,
        questions       = questions,
        k_range         = range(args.k_min, args.k_max + 1),
        use_wandb       = not args.no_wandb,
        wandb_project   = args.wandb_project,
        use_attention   = not args.no_attention,
        embedding_model = embedding_model,
        reranker_model  = reranker_model,
    )


if __name__ == "__main__":
    main()
