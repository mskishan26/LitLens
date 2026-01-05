import modal

LOCAL_CODE_DIR = "."
MODAL_REMOTE_CODE_DIR = "/root/app"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
        .uv_pip_install(
        "vllm",
        extra_options="--torch-backend=cu128"
    )
    .uv_pip_install(
        "sentence-transformers==5.2.0",
	    "chromadb==1.4.0",
        "nltk==3.9.2",
        "rank-bm25",
        "accelerate==1.12.0",
            )
    # might want to add boto3 if using AWS S3 for chat storage 
    .run_function(lambda: __import__("nltk").download("punkt_tab"))
    .add_local_dir(LOCAL_CODE_DIR,remote_path=MODAL_REMOTE_CODE_DIR)
    .env({"ENV": "prod", "HF_HOME": "/models"})
)

# i think gemini pro is using old methodology so we need to fix that

