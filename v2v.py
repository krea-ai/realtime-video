import cv2
import time
import torchvision.transforms.functional as TF
import torch
import numpy as np
import subprocess
from functools import lru_cache
import cv2
from PIL import Image
import requests
import json


def get_rotation_metadata(video_path):
    """
    Returns the rotation metadata (in degrees) from a video file using ffprobe.
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream_tags=rotate',
             '-of', 'json', video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        ffprobe_output = json.loads(result.stdout)
        tags = ffprobe_output.get('streams', [{}])[0].get('tags', {})
        rotation = int(tags.get('rotate', 0))
        return rotation
    except Exception as e:
        print(f"Warning: Could not read rotation metadata: {e}")
        return 0

def load_video_as_rgb(video_path, resample_to=None, resample_frame_count_threshold=81):
    import os
    import tempfile
    import urllib.request

    if video_path.startswith(('http://', 'https://')):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        try:
            # Use requests instead of urllib to handle potential 403 errors
            response = requests.get(video_path, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            video_path = temp_path
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise IOError(f"Failed to download video from URL: {e}")

    rotation = get_rotation_metadata(video_path)
    print("rotation", rotation)

    # Create a temporary file for the resampled video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as resampled_file:
        resampled_path = resampled_file.name
    
    try:
        # Use ffmpeg to resample the video to 16fps
        # Get the original frame count before resampling
        original_cap = cv2.VideoCapture(video_path)
        if not original_cap.isOpened():
            raise IOError(f"Cannot open original video file")
        
        original_frame_count = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = original_cap.get(cv2.CAP_PROP_FPS)
        original_cap.release()
        
        if resample_to is not None and original_frame_count > resample_frame_count_threshold:
            print(f"Original video: {original_frame_count} frames at {original_fps} fps")
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', video_path, 
                '-filter:v', f'fps=16',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '22',
                resampled_path
            ]
            
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            resampled_path = video_path
        # Now read the resampled video
        cap = cv2.VideoCapture(resampled_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open resampled video file")

        frames = []
        t = time.time() 
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # the ffmpeg resampling already rotates the video if needed
            if resample_to is None:
                # Rotate frame if needed
                if rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)

        cap.release()
        t_read = time.time() - t
        print("read time: ", t_read)
        if resample_to is not None:
            print(f"resampled to {resample_to}fps")
        print("Frame count: ", len(frames))

        
    finally:
        # Clean up temporary files
        if os.path.exists(resampled_path) and resampled_path != video_path:
            os.remove(resampled_path)
            
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

    frames = [TF.to_tensor(frame).sub_(0.5).div_(0.5) for frame in frames]
    return torch.stack(frames)

def get_denoising_schedule(timesteps, denoising_strength, steps = 4):
    denoising_step_list = torch.linspace(denoising_strength * 1000, 0, steps, dtype=torch.float32, device=timesteps.device).to(torch.long)
    denoising_step_list = timesteps[1000 - denoising_step_list]
    return denoising_step_list

def encode_video_latent(vae, encode_vae_cache, resample_to=16, max_frames=81, video_path_or_url=None, frames=None, height=None, width=None, stream=False, dtype=torch.float16):
    vae_stride = (4, 8, 8)
    if frames is None:
        frames = load_video_as_rgb(video_path_or_url, resample_to=resample_to, resample_frame_count_threshold=33)
    h, w = frames.shape[2:] if (height is None and width is None) else height, width

    if max_frames is None:
        max_frames = 1 + ((frames.shape[0] - 1) // 4) * 4
    if max_frames:
        frames = frames[:max_frames]
    print("frames shape: ", frames.shape)

    h = h // vae_stride[1] * vae_stride[1]
    w = w // vae_stride[2] * vae_stride[2]

    frames = torch.nn.functional.interpolate(frames.cuda(), size=(h, w), mode='bicubic').transpose(0, 1).to(dtype)
    
    init_video_latents, encode_vae_cache = vae(frames.unsqueeze(0), encode_vae_cache, stream=stream)
    del frames

    return init_video_latents.squeeze(0).to(dtype), encode_vae_cache

if __name__ == "__main__":
    video_path = "https://app-uploads.krea.ai/public/a8218957-1a80-43dc-81b2-da970b5f2221-video.mp4"
    x = load_video_as_rgb(video_path, resample_to=16)

