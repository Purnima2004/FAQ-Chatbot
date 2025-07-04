from huggingface_hub import snapshot_download
import os

def download_models():
    """Download required models and cache them locally."""
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    
    print("Downloading sentence-transformers/all-MiniLM-L6-v2...")
    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        repo_type="model",
        local_dir=os.path.join(models_dir, "sentence-transformers_all-MiniLM-L6-v2")
    )
    
    print("\nDownloading google/flan-t5-base...")
    snapshot_download(
        repo_id="google/flan-t5-base",
        repo_type="model",
        local_dir=os.path.join(models_dir, "google-flan-t5-base")
    )
    
    print("\nAll models downloaded successfully!")

if __name__ == "__main__":
    download_models()
