import pandas as pd
import wandb
import os

def upload_csv_to_wandb(csv_path, project_name="rag-ablation-study"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return

    # Initialize wandb
    run = wandb.init(project=project_name, name=f"recovered-run-{os.path.basename(csv_path)}")
    
    # Read raw data
    df = pd.read_csv(csv_path)
    
    # Sort by k to ensure lines in wandb make sense
    df = df.sort_values(by=['k'])
    
    # Log each row as a step
    for _, row in df.iterrows():
        log_dict = {
            "k": row['k'],
            "strategy": row['strategy'],
            "hit_ratio": row['top1_hit_ratio'],
            "ttft": row['ttft'],
            "tps": row['tps'],
            "answer_len": row['answer_len']
        }
        # Create a combined metric for easier comparison in charts
        prefix = f"{row['strategy']}_k{row['k']}"
        wandb.log(log_dict)

    run.finish()

if __name__ == "__main__":
    # Upload the specific raw results from the 17:52:51 run
    upload_csv_to_wandb("experiment_results/raw_results_20260308_175251.csv")
