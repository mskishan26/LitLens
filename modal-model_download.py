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
def sync_models(model_dict: dict):
    from huggingface_hub import snapshot_download
    import os

    for model in model_dict.values():
        print(f"Checking/Downloading: {model['id']}")
        target_dir = model['path']
        
        # snapshot_download automatically checks if files exist
        snapshot_download(
            repo_id=model['id'], 
            local_dir=target_dir,
            revision=model['revision'],
            # This ensures we don't redownload everything if it's already there
            local_files_only=False 
        )
    
    # Crucial: Commit changes so they persist in the Volume
    volume.commit()
    print("All models synced to volume.")

# 3. The Orchestrator (Runs locally)
@app.local_entrypoint()
def model_download():
    try:
        from utils.config_loader import load_config
        config = load_config()
        model_names = config['models']
    except ImportError as e:
        print(f"Critical Error: Could not find 'utils.config_loader'. Ensure the folder exists. {e}")
        raise # Stops execution and alerts Modal/CI-CD
    except FileNotFoundError as e:
        print(f"Critical Error: Configuration file missing. {e}")
        raise
    except KeyError as e:
        print(f"Critical Error: Your config is missing the expected key: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
    print(f"Found {len(model_names)} models in config. Starting sync...")
    
    # Trigger the remote Modal function
    sync_models.remote(model_names) 
    
@app.local_entrypoint()
def upload_data():
    from utils.config_loader import load_config
    config = load_config()
    data_root = Path(config['paths']['data_root'])
    
    # Define which sub-folders to sync
    targets = ["logs", "embeddings", "bm25_artifacts"]
    
    data_vol = modal.Volume.from_name("data-storage-vol", create_if_missing=True)
    
    with data_vol.batch_upload() as batch:
        for folder in targets:
            local_path = data_root / folder
            if local_path.exists():
                print(f"Uploading {folder}...")
                batch.put_directory(local_path, remote_path=f"/{folder}")
    print("Data sync complete.")