from huggingface_hub import snapshot_download
snapshot_download(repo_id="Undi95/Llama-3-LewdPlay-8B-evo", ignore_patterns=["*.gguf"])  # Download our BF16 model without downloading GGUF models.


