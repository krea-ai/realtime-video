"""
Modal deployment for Krea Realtime Video with Webcam Support
Run with: modal deploy modal_app.py
"""

import modal
import os
import shutil
from pathlib import Path

# Create Modal app
app = modal.App("krea-realtime-video")

# Create persistent volume for model caching
models_volume = modal.Volume.from_name("krea-models", create_if_missing=True)

# Define the container image with all dependencies
# Use NVIDIA CUDA devel image to get nvcc and CUDA toolkit for flash-attn
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg")
    # Install PyTorch 2.8 from CUDA 12.8 index (includes B200 sm_100 support)
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        index_url="https://download.pytorch.org/whl/cu128"
    )
    # Install build dependencies for flash-attn
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    # Install latest Flash Attention with Blackwell (B200) support
    .pip_install("flash-attn==2.8.3", extra_options="--no-build-isolation")
    # Install CLIP from git
    .pip_install("git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1")
    # Install core dependencies
    .pip_install(
        "accelerate==1.9.0",
        "diffusers==0.31.0",
        "easydict==1.13",
        "einops==0.8.1",
        "fastapi==0.116.1",
        "huggingface-hub==0.34.3",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "msgpack==1.1.1",
        "numpy==2.1.2",
        "omegaconf==2.3.0",
        "open-clip-torch==3.0.0",
        "opencv-python==4.11.0.86",
        "pillow==10.4.0",
        "pydantic==2.10.6",
        "python-dotenv==1.1.1",
        "python-multipart>=0.0.20",
        "safetensors==0.5.3",
        "sentencepiece==0.2.0",
        "timm==1.0.19",
        "transformers==4.54.1",
        "uvicorn==0.35.0",
    )
    # Background removal (RF-DETR and YOLOv8)
    .pip_install(
        "ultralytics==8.3.29",
        "roboflow==1.2.11",
    )
    # Add local source code to the image (use script directory, not cwd)
    .add_local_dir(str(Path(__file__).parent), "/root/app")
)

# Download models on container build
@app.function(
    image=image,
    volumes={"/models": models_volume},
    timeout=3600,
)
def download_models():
    """Download required models to persistent volume"""
    import subprocess

    models_dir = "/models"
    os.makedirs(models_dir, exist_ok=True)

    # Download BOTH base models (checkpoint auto-detects which one to use)
    # 1.3B base
    base_1_3b_path = f"{models_dir}/Wan2.1-T2V-1.3B"
    if not os.path.exists(base_1_3b_path):
        print("Downloading Wan 2.1 1.3B base model...")
        subprocess.run([
            "huggingface-cli", "download",
            "Wan-AI/Wan2.1-T2V-1.3B",
            "--local-dir-use-symlinks", "False",
            "--local-dir", base_1_3b_path
        ], check=True)

    # 14B base (krea-realtime-video-14b.safetensors needs this)
    base_14b_path = f"{models_dir}/Wan2.1-T2V-14B"
    if not os.path.exists(base_14b_path):
        print("Downloading Wan 2.1 14B base model...")
        subprocess.run([
            "huggingface-cli", "download",
            "Wan-AI/Wan2.1-T2V-14B",
            "--local-dir-use-symlinks", "False",
            "--local-dir", base_14b_path
        ], check=True)

    # Download checkpoints
    checkpoint_dir = f"{models_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Krea Realtime 14B checkpoint
    checkpoint_14b_path = f"{checkpoint_dir}/krea-realtime-video-14b.safetensors"
    if not os.path.exists(checkpoint_14b_path):
        print("Downloading Krea Realtime 14B checkpoint...")
        subprocess.run([
            "huggingface-cli", "download",
            "krea/krea-realtime-video",
            "krea-realtime-video-14b.safetensors",
            "--local-dir", checkpoint_dir
        ], check=True)

    # 2. Self-Forcing 1.3B checkpoint
    checkpoint_13b_path = f"{checkpoint_dir}/self_forcing_dmd.pt"
    if not os.path.exists(checkpoint_13b_path):
        print("Downloading Self-Forcing 1.3B checkpoint...")
        # Download to a temp location and move it to flatten the path
        temp_checkpoint_dir = f"{models_dir}/self_forcing_temp"
        subprocess.run([
            "huggingface-cli", "download",
            "gdhe17/Self-Forcing",
            "checkpoints/self_forcing_dmd.pt",
            "--local-dir", temp_checkpoint_dir
        ], check=True)
        # Move from nested path to flat path
        nested_file = f"{temp_checkpoint_dir}/checkpoints/self_forcing_dmd.pt"
        if os.path.exists(nested_file):
            shutil.move(nested_file, checkpoint_13b_path)
            # Clean up temp directory
            shutil.rmtree(temp_checkpoint_dir, ignore_errors=True)

    models_volume.commit()
    print("✓ Both 14B and 1.3B models downloaded and cached!")
    return True


# Main web server
with image.imports():
    import sys
    import os

@app.function(
    image=image,
    # B200 is the OPTIMAL GPU for this model (HBM3e memory bandwidth)
    gpu="B200",  # 11 fps (verified by Krea), $6.25/hour - FASTEST for this model
    # Alternatives if B200 unavailable or too expensive:
    # gpu=modal.gpu.H200(),  # Likely ~8-10 fps, $4.54/hour
    # gpu=modal.gpu.H100(count=1),  # Likely ~7-9 fps, $3.95/hour (memory bottleneck)
    # gpu=modal.gpu.A100(count=1, size="80GB"),  # Likely ~6-8 fps, $2.50/hour
    volumes={"/models": models_volume},
    timeout=28800,  # 8 hours for long sessions
    allow_concurrent_inputs=10,
    container_idle_timeout=300,
)
@modal.asgi_app()
def serve():
    """Serve the FastAPI application with webcam support"""
    import sys
    import os

    # Set up environment
    os.environ["MODEL_FOLDER"] = "/models"
    os.environ["DO_COMPILE"] = "true"

    # Change to app directory
    os.chdir("/root/app")
    sys.path.insert(0, "/root/app")

    # Select model based on MODEL_VERSION env var (default: 14b)
    model_version = os.environ.get("MODEL_VERSION", "14b").lower()
    if model_version == "1.3b":
        config_file = "/root/app/configs/self_forcing_server.yaml"
        print("✓ Using 1.3B model (Self-Forcing)")
    else:
        config_file = "/root/app/configs/self_forcing_server_14b.yaml"
        print("✓ Using 14B model (Krea Realtime)")

    os.environ["CONFIG"] = config_file

    # Create symlink for checkpoints
    if not os.path.exists("/root/app/checkpoints"):
        os.symlink("/models/checkpoints", "/root/app/checkpoints")

    # Import and return the FastAPI app
    from release_server import app as fastapi_app

    return fastapi_app


@app.local_entrypoint()
def main():
    """Download models before deploying"""
    print("Downloading models...")
    download_models.remote()
    print("Models ready! Deploy with: modal deploy modal_app.py")
