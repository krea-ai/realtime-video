import os
import asyncio
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

from release_server import (
    load_merge_config, 
    load_all, 
    GenerateParams, 
    GenerationSession,
    Models
)

torch.set_grad_enabled(False)

# Default prompts
prompts = [
    "A person is running in a park",
    "A person is running in a forest",
]


def save_video_direct(pixels: torch.Tensor, output_path: Path, fps: int = 30):
    """Save video directly without intermediate frames using torchvision"""
    try:
        import torchvision.io as io
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # pixels shape: [1, num_frames, 3, H, W] -> [num_frames, H, W, 3]
        video_tensor = pixels[0].permute(0, 2, 3, 1).cpu().clamp(0, 1)
        # Convert to uint8 [0, 255]
        video_tensor = (video_tensor * 255).to(torch.uint8)
        # torchvision expects [T, H, W, C]
        
        io.write_video(
            str(output_path),
            video_tensor,
            fps=fps,
            video_codec='h264',
            options={'crf': '18'}  # High quality
        )
        print(f"✅ Video saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error saving video: {e}")
        # Fallback to ffmpeg method if torchvision fails
        return save_video_ffmpeg_pipe(pixels, output_path, fps)


def save_video_ffmpeg_pipe(pixels: torch.Tensor, output_path: Path, fps: int = 30):
    """Fallback: Save video using ffmpeg pipe (no intermediate files)"""
    import subprocess
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # pixels shape: [1, num_frames, 3, H, W]
    video_tensor = pixels[0].cpu().clamp(0, 1)
    num_frames, _, height, width = video_tensor.shape
    
    # Convert to uint8 RGB frames
    video_np = (video_tensor.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",  # Read from stdin
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",  # High quality
        "-preset", "fast",
        str(output_path)
    ]
    
    try:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        process.stdin.write(video_np.tobytes())
        process.stdin.close()
        process.wait()
        
        if process.returncode == 0:
            print(f"✅ Video saved to {output_path}")
            return output_path
        else:
            print(f"❌ Error creating video: {process.stderr.read().decode()}")
            return None
    except Exception as e:
        print(f"❌ Error creating video: {e}")
        return None
    

def save_video_frames(pixels: torch.Tensor, output_dir: Path, prompt_idx: int):
    """Save video frames as images"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # pixels shape: [1, num_frames, 3, H, W]
    num_frames = pixels.shape[1]
    pixels = pixels[0]  # Remove batch dimension
    
    saved_frames = []
    for frame_idx in range(num_frames):
        frame = pixels[frame_idx]  # [3, H, W]
        # Convert to PIL image
        pil_image = TF.to_pil_image(frame.cpu().clamp(0, 1))
        
        # Save frame
        frame_path = output_dir / f"prompt_{prompt_idx:03d}_frame_{frame_idx:04d}.png"
        pil_image.save(frame_path)
        saved_frames.append(frame_path)
    
    return saved_frames


def create_video_from_frames(frames_dir: Path, output_path: Path, fps: int = 30):
    """Create a video from frames using ffmpeg"""
    import subprocess
    
    # Get all frame files for this prompt
    frame_pattern = str(frames_dir / "*.png")
    
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Video saved to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating video: {e.stderr.decode()}")
        return None


def sample_videos(
    prompts_list,
    config_path: str = "configs/self_forcing_server_14b.yaml",
    output_dir: str = "outputs/samples",
    params: GenerateParams = None,
    models: Models = None,
    save_videos: bool = True,
    fps: int = 16,
):
    """
    Sample videos for a list of prompts.
    
    Args:
        prompts_list: List of text prompts
        config_path: Path to config file
        output_dir: Directory to save outputs
        num_blocks: Number of blocks to generate (default: 15)
        width: Video width (default: 832)
        height: Video height (default: 480)
        seed: Random seed (default: 42)
        is_t2v: Text-to-video mode (default: True)
        models: Pre-loaded models (if None, will load from config)
        save_videos: Whether to save as video files (requires ffmpeg)
        fps: Frames per second for video output
        
    Returns:
        dict: Results containing frame paths and video paths for each prompt
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load models once and reuse
    load_models = models is None
    if load_models:
        print("🔄 Loading models...")
        config = load_merge_config(config_path)
        models = load_all(config)
        print("✅ Models loaded successfully")
    else:
        # Get config from existing models
        config = load_merge_config(config_path)
    
    results = {}
    
    # Sample for each prompt
    for prompt_idx, prompt in enumerate(tqdm(prompts_list, desc="Sampling videos")):
        print(f"\n📝 Prompt {prompt_idx + 1}/{len(prompts_list)}: {prompt}")
        
        
        # Callback to collect frames
        all_frames = []
        def frame_callback(pixels, frame_ids, event):
            # Wait for GPU computation to finish
            event.synchronize()
            # pixels: [1, num_frames, 3, H, W] on GPU
            cpu_pixels = pixels.cpu().add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
            all_frames.append(cpu_pixels)
        
        params.prompt = prompt
        # Create generation session
        session = GenerationSession(
            params=params,
            config=config,
            frame_callback=frame_callback,
            models=models
        )
        
        # Generate all blocks
        num_blocks = params.num_blocks
        print(f"🎬 Generating {num_blocks} blocks...")
        t_start_sampling = time.time() 
        for block_idx in range(num_blocks):
            try:
                session.generate_block(models)
            except asyncio.CancelledError:
                break
                
        
        # Combine all frames
        combined_frames = torch.cat(all_frames, dim=1)
        print(f"✅ Generated {combined_frames.shape[1]} frames")
        t_end_sampling = time.time()
        print(f"Sampling took {t_end_sampling - t_start_sampling:.2f}s")
        
        result = {
            "prompt": prompt,
            "num_frames": combined_frames.shape[1],
            "video_path": None,
        }
        
        # Save video directly if requested
        if save_videos:
            video_path = output_path / f"prompt_{prompt_idx:03d}.mp4"
            video_result = save_video_direct(combined_frames, video_path, fps)
            result["video_path"] = video_result
        
        results[prompt_idx] = result
        
        # Clean up session
        session.dispose()
    
    print("\n🎉 All videos generated successfully!")
    return results


def create_grid(outputs_folder: str = "outputs", grid_output_dir: str = "outputs/grids"):
    """
    Create grid videos for each prompt from multiple output folders.
    
    Args:
        outputs_folder: Path to folder containing output subdirectories
        grid_output_dir: Directory to save grid videos
        
    Returns:
        dict: Mapping of prompt names to grid video paths
    """
    import subprocess
    from collections import defaultdict
    
    outputs_path = Path(outputs_folder)
    grid_path = Path(grid_output_dir)
    grid_path.mkdir(parents=True, exist_ok=True)
    
    if not outputs_path.exists():
        print(f"❌ Error: {outputs_folder} does not exist")
        return {}
    
    # Collect all video files grouped by prompt
    prompt_videos = defaultdict(list)  # prompt_name -> [(folder_name, video_path), ...]
    
    # Scan all subdirectories in outputs
    subdirs = [d for d in outputs_path.iterdir() if d.is_dir()]
    print(f"📁 Found {len(subdirs)} output folders: {[d.name for d in subdirs]}")
    
    for subdir in sorted(subdirs):
        folder_name = subdir.name
        # Find all .mp4 files in this folder
        video_files = sorted(subdir.glob("*.mp4"))
        
        for video_file in video_files:
            # Extract prompt name (e.g., "prompt_009" from "prompt_009.mp4")
            prompt_name = video_file.stem
            prompt_videos[prompt_name].append((folder_name, video_file))
    
    print(f"🎬 Found {len(prompt_videos)} unique prompts")
    
    grid_results = {}
    
    # Create grid for each prompt
    for prompt_name, videos in sorted(prompt_videos.items()):
        print(f"\n📹 Creating grid for {prompt_name} ({len(videos)} videos)")
        
        if len(videos) == 0:
            continue
        
        # Sort videos by folder name for consistent ordering
        videos = sorted(videos, key=lambda x: x[0])
        
        # Create a grid video using ffmpeg
        grid_output_path = grid_path / f"{prompt_name}_grid.mp4"
        
        # Build ffmpeg filter for grid with text labels
        num_videos = len(videos)
        
        # Calculate grid dimensions (prefer wider grids)
        import math
        cols = min(3, num_videos)  # Max 3 columns
        rows = math.ceil(num_videos / cols)
        
        # Build ffmpeg command
        cmd = ["ffmpeg", "-y"]
        
        # Add input files
        for folder_name, video_path in videos:
            cmd.extend(["-i", str(video_path)])
        
        # Get dimensions from first video
        import subprocess as sp
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0",
            str(videos[0][1])
        ]
        try:
            dimensions = sp.run(probe_cmd, capture_output=True, text=True, check=True).stdout.strip()
            video_width, video_height = map(int, dimensions.split('x'))
        except:
            # Fallback to default dimensions if probe fails
            video_width, video_height = 832, 480
        
        # Build filter complex for grid layout with text
        filter_parts = []
        
        # Add text overlay to each video showing folder name
        for i, (folder_name, _) in enumerate(videos):
            import re

            folder_name = re.sub(r"_([0-9]+(?:\.[0-9]+)?)", lambda m: f"_{float(m.group(1)):.2f}", folder_name)

            # Add text overlay with folder name (no scaling)
            filter_parts.append(
                f"[{i}:v]drawtext=text='{folder_name}':"
                f"fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5:"
                f"boxborderw=5:x=(w-text_w)/2:y=h-40[v{i}]"
            )
        
        # Create grid layout using xstack
        if num_videos == 1:
            layout = "0_0"
            inputs = "[v0]"
        else:
            # Build xstack layout
            layout_positions = []
            inputs = "".join(f"[v{i}]" for i in range(num_videos))
            
            for i in range(num_videos):
                row = i // cols
                col = i % cols
                x = col * video_width
                y = row * video_height
                layout_positions.append(f"{x}_{y}")
            
            layout = "|".join(layout_positions)
        
        filter_complex = ";".join(filter_parts)
        if num_videos > 1:
            filter_complex += f";{inputs}xstack=inputs={num_videos}:layout={layout}[outv]"
            output_map = "[outv]"
        else:
            output_map = "[v0]"
        
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", output_map,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            str(grid_output_path)
        ])
        
        # Run ffmpeg
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Grid video saved to {grid_output_path}")
            grid_results[prompt_name] = grid_output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating grid for {prompt_name}:")
            print(f"   Command: {' '.join(cmd)}")
            print(f"   Error: {e.stderr}")
    
    print(f"\n🎉 Created {len(grid_results)} grid videos in {grid_output_dir}")
    return grid_results


def sample_single_video(
    prompt: str,
    config_path: str = "configs/self_forcing_dmd_will_optims.yaml",
    output_path: str = "output_video.mp4",
    num_blocks: int = 15,
    width: int = 832,
    height: int = 480,
    seed: int = 42,
    models: Models = None,
):
    """
    Sample a single video for a prompt.
    
    Args:
        prompt: Text prompt
        config_path: Path to config file
        output_path: Path to save output video
        num_blocks: Number of blocks to generate (default: 15)
        width: Video width (default: 832)
        height: Video height (default: 480) 
        seed: Random seed (default: 42)
        models: Pre-loaded models (if None, will load from config)
        
    Returns:
        Path to generated video
    """
    result = sample_videos(
        prompts_list=[prompt],
        config_path=config_path,
        output_dir=str(Path(output_path).parent),
        width=width,
        height=height,
        seed=seed,
        models=models,
        save_videos=True,
    )
    
    return result[0]["video_path"]


# Main execution
if __name__ == "__main__":
    # Sample videos for all prompts
    results = sample_videos(
        prompts_list=prompts,
        num_blocks=15,
        is_t2v=True,
    )
