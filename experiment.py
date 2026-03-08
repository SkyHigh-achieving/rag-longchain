import os
import sys
import time
import math
import csv
import argparse
import logging
import re
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("WandB not installed. Run 'pip install wandb' to enable tracking.")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.processor import DocumentProcessor
from core.retriever import AdvancedRetriever
from core.generator import Generator

# --- Helper Functions (Adapted from app.py) ---

def _normalize_for_overlap(text):
    if not text:
        return []
    # Simple tokenization for Jaccard similarity
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE).lower()
    return [t for t in text.split() if len(t) >= 2]

def _top1_hit(answer, top1_text):
    ans_tokens = set(_normalize_for_overlap(answer))
    doc_tokens = set(_normalize_for_overlap(top1_text))
    if not ans_tokens or not doc_tokens:
        return False, 0.0
    inter = len(ans_tokens & doc_tokens)
    ratio = inter / max(len(ans_tokens), 1)
    # Threshold 0.12 from app.py
    return ratio >= 0.12, ratio

def _measure_perf(generator, query, context_docs):
    start = time.perf_counter()
    first_token_ts = None
    chunks = []
    
    # Use generator's stream method
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

# --- Experiment Runner ---

class ExperimentRunner:
    def __init__(self, pdf_path, questions, use_wandb=True, k_max=6, no_attention=False, compare_hw=False):
        self.pdf_path = pdf_path
        self.questions = questions
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.k_max = k_max
        self.no_attention = no_attention
        self.compare_hw = compare_hw
        
        self.retriever = None
        self.generator = None
        
        # Data storage
        self.raw_results = []
        self.agg_results = []

    @staticmethod
    def _set_hardware_mode(hw_mode):
        mode = (hw_mode or "auto").lower()
        if mode == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            return "cpu"
        if mode == "gpu":
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            return "gpu"
        return "auto"

    def setup(self):
        logger.info(f"Loading PDF from {self.pdf_path}...")
        processor = DocumentProcessor()
        if not os.path.exists(self.pdf_path):
            # Try to create dummy if missing
            logger.warning(f"PDF not found at {self.pdf_path}. Attempting to generate demo PDF...")
            try:
                import subprocess
                subprocess.run([sys.executable, "create_dummy_pdf.py"], check=True)
                if not os.path.exists(self.pdf_path):
                    raise FileNotFoundError("Failed to generate demo PDF.")
            except Exception as e:
                logger.error(f"Could not generate PDF: {e}")
                raise

        docs = processor.load_pdf(self.pdf_path)
        logger.info(f"Loaded {len(docs)} document chunks.")
        
        logger.info("Initializing AdvancedRetriever...")
        self.retriever = AdvancedRetriever(docs)
        
        logger.info("Initializing Generator...")
        self.generator = Generator()
        
        if self.use_wandb:
            import torch
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
            wandb.init(
                project="rag-ablation-study",
                name=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "k_max": self.k_max,
                    "model": "Local-HF" if self.generator.use_local_hf else "API",
                    "pdf": os.path.basename(self.pdf_path),
                    "questions_count": len(self.questions),
                    "gpu": gpu_name,
                    "compare_hw": self.compare_hw
                }
            )

    def _refresh_generator(self, hw_mode):
        self._set_hardware_mode(hw_mode)
        self.generator = Generator()
        if self.generator.use_local_hf:
            device_text = str(getattr(getattr(self.generator, "local_model", None), "device", "")).lower()
            active = "HF-GPU" if "cuda" in device_text else "HF-CPU"
        else:
            active = "API"
        logger.info(f"Generator initialized for {hw_mode}: {active}")

    def run(self):
        logger.info("Starting Experiment Loop...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_csv_path = os.path.join("data", f"experiment_raw_{timestamp}.csv")
        agg_csv_path = os.path.join("data", f"experiment_agg_{timestamp}.csv")
        
        # CSV Headers
        raw_headers = [
            "k", "question_id", "strategy", "hardware",
            "ttft", "tps", "hit_ratio", "is_hit", 
            "concentration", "entropy", "error"
        ]
        
        with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(raw_headers)

        hw_modes = ["auto"]
        if self.compare_hw:
            if os.getenv("LOCAL_LLM_MODE", "").strip().lower() == "hf":
                hw_modes = ["gpu", "cpu"]
            else:
                logger.warning("compare-hw requires LOCAL_LLM_MODE=hf. Current mode is API-compatible; using AUTO only.")

        for k in range(1, self.k_max + 1):
            logger.info(f"Running for k={k}...")
            
            k_metrics = {}
            k_attempts = 0
            k_success = 0
            
            for hw in hw_modes:
                self._refresh_generator(hw)

                for mode in ["Advanced", "Standard"]:
                    strat_hw_key = f"{mode}_{hw.upper()}"
                    k_metrics[strat_hw_key] = {"hit": [], "tps": [], "ttft": [], "conc": [], "ent": []}

                    for q_idx, query in enumerate(self.questions):
                        k_attempts += 1
                        try:
                            # 1. Retrieval
                            if mode == "Advanced":
                                docs = self.retriever.retrieve_with_rerank(query, top_k=k)
                            else:
                                docs = self.retriever.vector_retriever.invoke(query)[:k]
                            
                            # 2. Generation & Perf
                            answer, ttft, tps = _measure_perf(self.generator, query, docs)
                            
                            # 3. Attention Analysis
                            concentration = 0.0
                            entropy = 0.0
                            top1_text = ""
                            
                            if not self.no_attention and self.generator.use_local_hf:
                                attn_res = self.generator.analyze_attention_struct(query, docs, top_n=min(3, k))
                                if attn_res.get("ok"):
                                    concentration = attn_res.get("concentration", 0.0)
                                    ranked = attn_res.get("all_scores", [])
                                    if ranked:
                                        entropy = -sum(item["weight"] * math.log(max(item["weight"], 1e-9)) for item in ranked)
                                    if attn_res.get("ranked"):
                                        top1_text = attn_res["ranked"][0]["text"]
                            
                            if not top1_text and docs:
                                top1_text = docs[0].page_content

                            # 4. Metrics
                            is_hit, hit_ratio = _top1_hit(answer, top1_text)
                            
                            # Log raw result
                            row = [
                                k, q_idx, mode, hw,
                                f"{ttft:.4f}", f"{tps:.2f}", f"{hit_ratio:.4f}", is_hit, 
                                f"{concentration:.4f}", f"{entropy:.4f}", ""
                            ]
                            
                            with open(raw_csv_path, "a", newline="", encoding="utf-8") as f:
                                csv.writer(f).writerow(row)
                                
                            # Aggregate
                            k_metrics[strat_hw_key]["hit"].append(hit_ratio)
                            k_metrics[strat_hw_key]["tps"].append(tps)
                            k_metrics[strat_hw_key]["ttft"].append(ttft)
                            if self.generator.use_local_hf and not self.no_attention:
                                k_metrics[strat_hw_key]["conc"].append(concentration)
                                k_metrics[strat_hw_key]["ent"].append(entropy)
                            k_success += 1

                        except Exception as e:
                            logger.error(f"Error k={k} q={q_idx} mode={mode} hw={hw}: {e}")
                            row = [k, q_idx, mode, hw, "", "", "", "", "", "", str(e)]
                            with open(raw_csv_path, "a", newline="", encoding="utf-8") as f:
                                csv.writer(f).writerow(row)

            # --- Aggregation per K ---
            def safe_mean(lst): return np.mean(lst) if lst else 0.0
            
            log_dict = {"k": k}
            for key, m in k_metrics.items():
                prefix = f"agg/{key.lower()}"
                log_dict.update({
                    f"{prefix}/hit_mean": safe_mean(m["hit"]),
                    f"{prefix}/tps_mean": safe_mean(m["tps"]),
                    f"{prefix}/ttft_mean": safe_mean(m["ttft"]),
                })
                if self.generator.use_local_hf and not self.no_attention:
                    log_dict.update({
                        f"{prefix}/concentration": safe_mean(m["conc"]),
                        f"{prefix}/entropy": safe_mean(m["ent"]),
                    })

            if "Advanced_AUTO" in k_metrics and "Standard_AUTO" in k_metrics:
                log_dict["agg/delta/hit_ratio"] = safe_mean(k_metrics["Advanced_AUTO"]["hit"]) - safe_mean(k_metrics["Standard_AUTO"]["hit"])
                log_dict["agg/delta/tps"] = safe_mean(k_metrics["Advanced_AUTO"]["tps"]) - safe_mean(k_metrics["Standard_AUTO"]["tps"])
            if "Advanced_GPU" in k_metrics and "Advanced_CPU" in k_metrics:
                gpu_tps = safe_mean(k_metrics["Advanced_GPU"]["tps"])
                cpu_tps = safe_mean(k_metrics["Advanced_CPU"]["tps"])
                log_dict["agg/hw/advanced_tps_gain"] = gpu_tps - cpu_tps
                log_dict["agg/hw/advanced_tps_ratio"] = (gpu_tps / max(cpu_tps, 1e-6))
            if "Standard_GPU" in k_metrics and "Standard_CPU" in k_metrics:
                gpu_tps = safe_mean(k_metrics["Standard_GPU"]["tps"])
                cpu_tps = safe_mean(k_metrics["Standard_CPU"]["tps"])
                log_dict["agg/hw/standard_tps_gain"] = gpu_tps - cpu_tps
                log_dict["agg/hw/standard_tps_ratio"] = (gpu_tps / max(cpu_tps, 1e-6))

            if k_success == 0:
                logger.error(f"K={k} has zero successful samples. Check raw csv error column.")
            
            # WandB Log
            if self.use_wandb:
                wandb.log(log_dict)
            
            # Local Agg Log
            self.agg_results.append(log_dict)
            logger.info(f"K={k} Aggregation Complete. success={k_success}/{k_attempts}, keys={len(log_dict)}")

        # Save Aggregated CSV
        if self.agg_results:
            keys = self.agg_results[0].keys()
            with open(agg_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.agg_results)
        
        logger.info(f"Experiment Finished. Results saved to {agg_csv_path}")
        if self.use_wandb:
            wandb.finish()

# --- Main Entry ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Ablation Study")
    parser.add_argument("--k-max", type=int, default=6, help="Max number of chunks to retrieve")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--no-attention", action="store_true", help="Skip attention analysis (faster)")
    parser.add_argument("--pdf-path", type=str, default="data/demo_paper.pdf", help="Path to PDF file")
    parser.add_argument("--compare-hw", action="store_true", help="Compare CPU vs GPU performance")
    
    args = parser.parse_args()
    
    # Define Questions (Based on demo_paper.pdf content)
    QUESTIONS = [
        "What architecture does this paper propose?",
        "What are the specific weights for BM25 and Vector Search?",
        "How much does the system outperform the baseline in recall?",
        "What dataset was used for the experimental results?",
        "What is mentioned as future work?",
        "Does the system use a Cross-Encoder?",
        "Why is exact keyword matching important according to the introduction?",
        "What is the role of BGE-Reranker in the methodology?",
        "Which retrieval method is considered the baseline?",
        "What domain is this architecture designed for?"
    ]
    
    runner = ExperimentRunner(
        pdf_path=args.pdf_path,
        questions=QUESTIONS,
        use_wandb=not args.no_wandb,
        k_max=args.k_max,
        no_attention=args.no_attention,
        compare_hw=args.compare_hw
    )
    
    try:
        runner.setup()
        runner.run()
    except Exception as e:
        logger.error(f"Fatal Error: {e}")
        sys.exit(1)
