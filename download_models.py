import os

# Create local directories
project_root = os.getcwd()
models_dir = os.path.join(project_root, "models")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Redirect ALL common user folder variables to project models folder
os.environ['MODELSCOPE_CACHE'] = models_dir
os.environ['MODELSCOPE_HOME'] = models_dir
os.environ['MODELSCOPE_SDK_HOME'] = models_dir
os.environ['USERPROFILE'] = models_dir
os.environ['HOMEDRIVE'] = 'D:'
os.environ['HOMEPATH'] = models_dir.replace('D:', '')

from modelscope import snapshot_download

print(f"Downloading models to {models_dir}...")

def download_with_fallback(model_ids, local_dir_name):
    for model_id in model_ids:
        try:
            print(f"Trying to download {model_id}...")
            snapshot_download(
                model_id,
                local_dir=os.path.join(models_dir, local_dir_name)
            )
            print(f"Successfully downloaded {model_id}!")
            return True
        except Exception as e:
            print(f"Failed to download {model_id}: {e}")
    return False

# 1. Embedding Model
if not os.path.exists(os.path.join(models_dir, 'bge-small-en-v1.5')):
    download_with_fallback(['AI-ModelScope/bge-small-en-v1.5', 'BAAI/bge-small-en-v1.5'], 'bge-small-en-v1.5')
else:
    print("bge-small-en-v1.5 already exists.")

# 2. Reranker Model
if not os.path.exists(os.path.join(models_dir, 'bge-reranker-base')):
    # Try multiple possible IDs for the reranker
    reranker_ids = [
        'AI-ModelScope/bge-reranker-base', 
        'BAAI/bge-reranker-base',
        'ZhipuAI/bge-reranker-base',
        'damo/nlp_corom_sentence-embedding_chinese-base-ecom' # fallback to another reranker if needed
    ]
    download_with_fallback(reranker_ids, 'bge-reranker-base')
else:
    print("bge-reranker-base already exists.")

print("Download process completed.")
