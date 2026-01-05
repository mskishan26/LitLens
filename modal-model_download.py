from typing import Optional
from pathlib import Path
import modal

# Setup the volume and image for model downloading
volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
MODEL_DIR = Path("/models")

download_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub[hf_transfer]") 
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("model-sync-manager")

@app.function(
    volumes={MODEL_DIR.as_posix(): volume},
    image=download_image,
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface")]
)
def sync_models(model_list: list[str]):
    from huggingface_hub import snapshot_download
    import os

    for repo_id in model_list:
        print(f"Checking/Downloading: {repo_id}")
        target_dir = MODEL_DIR / repo_id
        
        # snapshot_download automatically checks if files exist
        snapshot_download(
            repo_id=repo_id, 
            local_dir=target_dir,
            # This ensures we don't redownload everything if it's already there
            local_files_only=False 
        )
    
    # Crucial: Commit changes so they persist in the Volume
    volume.commit()
    print("All models synced to volume.")

# 3. The Orchestrator (Runs locally)
@app.local_entrypoint()
def main():
    try:
        from utils.config_loader import load_config
        config = load_config()
        model_names = list(config['models'].values())
    except ImportError:
        # Fallback for testing SHOULD I KEEP THIS SINCE MY GITHUB ACTIONS AS TO TECHNICALLY THROW AN ERROR AND GRACEFULLY HANDLE IT
        model_names = ["hf-internal-testing/tiny-random-GPTNeoXForCausalLM"]

    print(f"Found {len(model_names)} models in config. Starting sync...")
    
    # Trigger the remote Modal function
    sync_models.remote(model_names)