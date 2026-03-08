import argparse
import os
import shutil
import subprocess
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

os.environ["MODELSCOPE_CACHE"] = models_dir
os.environ["MODELSCOPE_HOME"] = models_dir
os.environ["MODELSCOPE_SDK_HOME"] = models_dir
os.environ["HOME"] = project_root
os.environ["USERPROFILE"] = project_root
os.environ["HF_HOME"] = models_dir
os.environ["TRANSFORMERS_CACHE"] = models_dir
os.environ["SENTENCE_TRANSFORMERS_HOME"] = models_dir

try:
    from modelscope import snapshot_download
except ImportError:
    print("缺少 modelscope，请先执行: .\\.venv\\Scripts\\python.exe -m pip install modelscope")
    sys.exit(1)


def download_with_fallback(model_ids, local_dir_name):
    target_dir = os.path.join(models_dir, local_dir_name)
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        print(f"{local_dir_name} 已存在，跳过下载。")
        return True
    for model_id in model_ids:
        try:
            print(f"正在尝试下载 {model_id} -> {target_dir}")
            snapshot_download(model_id, local_dir=target_dir)
            print(f"下载成功: {model_id}")
            return True
        except Exception as e:
            print(f"下载失败: {model_id} | {e}")
    return False


def download_retrieval_models():
    ok_emb = download_with_fallback(
        ["AI-ModelScope/bge-small-en-v1.5", "BAAI/bge-small-en-v1.5"],
        "bge-small-en-v1.5",
    )
    ok_rerank = download_with_fallback(
        [
            "AI-ModelScope/bge-reranker-base",
            "BAAI/bge-reranker-base",
            "ZhipuAI/bge-reranker-base",
        ],
        "bge-reranker-base",
    )
    print("HuggingFace 直链（可手动下载）：")
    print("https://huggingface.co/BAAI/bge-small-en-v1.5")
    print("https://huggingface.co/BAAI/bge-reranker-base")
    return ok_emb and ok_rerank


def download_ollama_qwen(model_tag):
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        print("未检测到 ollama 命令。请先安装 Ollama: https://ollama.com/download")
        print("安装后执行：ollama serve")
        print(f"再执行：ollama pull {model_tag}")
        return False
    ollama_models_dir = os.path.join(models_dir, "ollama")
    os.makedirs(ollama_models_dir, exist_ok=True)
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = ollama_models_dir
    print(f"Ollama 模型目录: {ollama_models_dir}")
    print(f"开始拉取本地 Qwen: {model_tag}")
    result = subprocess.run([ollama_bin, "pull", model_tag], env=env, check=False)
    if result.returncode == 0:
        print("Ollama Qwen 下载完成。")
        print(f"请确保启动命令窗口也设置 OLLAMA_MODELS={ollama_models_dir}")
        return True
    print("Ollama Qwen 下载失败，请检查网络或模型名。")
    return False


def download_local_hf_model(model_id, local_dir_name):
    target_dir = os.path.join(models_dir, local_dir_name)
    config_path = os.path.join(target_dir, "config.json")
    if os.path.isfile(config_path):
        print(f"{local_dir_name} 已存在，跳过下载。")
        return True
    modelscope_candidates = [model_id, "Qwen/Qwen2.5-0.5B-Instruct", "qwen/Qwen2.5-0.5B-Instruct"]
    for candidate in modelscope_candidates:
        try:
            print(f"尝试 ModelScope 下载本地模型: {candidate}")
            snapshot_download(candidate, local_dir=target_dir)
            if os.path.isfile(config_path):
                print(f"ModelScope 下载成功: {candidate}")
                return True
        except Exception as e:
            print(f"ModelScope 下载失败: {candidate} | {e}")
    try:
        os.makedirs(target_dir, exist_ok=True)
        print(f"开始下载本地 HuggingFace 模型: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
        tokenizer.save_pretrained(target_dir)
        model.save_pretrained(target_dir)
        print(f"本地模型已保存到: {target_dir}")
        return True
    except Exception as e:
        print(f"本地 HuggingFace 模型下载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-ollama-qwen", action="store_true")
    parser.add_argument("--ollama-model", default="qwen2.5:7b")
    parser.add_argument("--with-local-hf", action="store_true")
    parser.add_argument("--local-model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--local-dir-name", default="local-qwen2.5-0.5b-instruct")
    args = parser.parse_args()

    print(f"模型下载目录: {models_dir}")
    retrieval_ok = download_retrieval_models()
    ollama_ok = True
    local_hf_ok = True
    if args.with_ollama_qwen:
        ollama_ok = download_ollama_qwen(args.ollama_model)
    if args.with_local_hf:
        local_hf_ok = download_local_hf_model(args.local_model_id, args.local_dir_name)
    if not args.with_ollama_qwen and not args.with_local_hf:
        print("当前为云端 API 模式：无需下载 Qwen 大模型权重到本地。")
        print("请在 .env 中配置 OPENAI_API_BASE / OPENAI_API_KEY / MODEL_NAME。")
        print("阿里云通义兼容端点示例：https://dashscope.aliyuncs.com/compatible-mode/v1")

    if retrieval_ok and ollama_ok and local_hf_ok:
        print("全部流程完成。")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
