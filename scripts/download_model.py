#!/usr/bin/env python
import argparse
import os
import sys
from huggingface_hub import snapshot_download
from tqdm import tqdm

def download_llava_model(output_dir="models"):
    """Download LLaVA-1.5-7B model weights from Hugging Face."""
    print("Downloading LLaVA-1.5-7B model weights...")
    model_id = "llava-hf/llava-1.5-7b-hf"
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=os.path.join(output_dir, "llava-1.5-7b"),
            local_dir_use_symlinks=False,
        )
        print(f"Successfully downloaded LLaVA model to {output_dir}/llava-1.5-7b")
    except Exception as e:
        print(f"Error downloading LLaVA model: {e}")
        sys.exit(1)

def download_qwen_model(output_dir="models"):
    """Download Qwen-VL-Chat-7B model weights from Hugging Face."""
    print("Downloading Qwen-VL-Chat-7B model weights...")
    model_id = "Qwen/Qwen-VL-Chat"
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=os.path.join(output_dir, "qwen-vl-chat-7b"),
            local_dir_use_symlinks=False,
        )
        print(f"Successfully downloaded Qwen-VL model to {output_dir}/qwen-vl-chat-7b")
    except Exception as e:
        print(f"Error downloading Qwen-VL model: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download model weights for Conscious AI Agent")
    parser.add_argument("--model", type=str, required=True, choices=["llava-1.5-7b", "qwen-vl-chat-7b"],
                        help="Model to download (llava-1.5-7b or qwen-vl-chat-7b)")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory to save model weights (default: models/)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.model == "llava-1.5-7b":
        download_llava_model(args.output_dir)
    elif args.model == "qwen-vl-chat-7b":
        download_qwen_model(args.output_dir)

if __name__ == "__main__":
    main()
