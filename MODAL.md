# Running Krea Realtime Video on Modal

Complete guide for deploying and using the webcam-enabled realtime video app on Modal with B200 GPU.

## Quick Start

Your app will be deployed at:
**https://YOUR_WORKSPACE--krea-realtime-video-serve.modal.run**

(Replace `YOUR_WORKSPACE` with your Modal workspace name)

## What You Have

- **GPU**: NVIDIA B200 (Blackwell) - $6.25/hour when active
- **Stack**:
  - PyTorch 2.8.0+cu128 (B200 sm_100 support)
  - Flash Attention 2.8.3 (latest, with Blackwell optimizations)
  - CUDA 12.8 runtime libraries
- **Models**: 107GB cached in Modal cloud storage
  - Wan2.1-T2V-1.3B base model
  - Wan2.1-T2V-14B base model
  - krea-realtime-video-14b.safetensors checkpoint
- **Features**:
  - Text-to-Video generation
  - Video-to-Video transformation
  - **Webcam real-time transformation** (PR #3)
- **Auto-shutdown**: After 5 minutes of inactivity

## Using the App

### Webcam Mode (Real-time Transformation)

1. Open: https://YOUR_WORKSPACE--krea-realtime-video-serve.modal.run
2. Click "**Webcam**" radio button
3. Allow camera access when prompted
4. **Adjust Settings**:
   - **Denoising Strength**: 0.9-0.95 (for strong transformation)
   - **Num Blocks**: 1000 (for continuous streaming)
   - **Webcam FPS**: 15-30
   - **Resolution**: 832×480 (default) or 640×480 (faster)
5. Enter a descriptive prompt:
   - `"anime style, Studio Ghibli, vibrant colors, hand-drawn animation"`
   - `"cyberpunk character, intense neon lights, rain, futuristic city, dramatic"`
   - `"watercolor painting, soft brushstrokes, pastel colors, artistic"`
6. Click "**Start Generation**"
7. Watch your webcam feed transform in real-time!

**Side-by-side view:**
- **Left**: Generated transformed output (~11 fps on B200)
- **Right**: Your original webcam feed

### Text-to-Video Mode

1. Keep "**Text-to-Video**" selected (default)
2. Enter prompt: `"ocean waves crashing at sunset, cinematic"`
3. Settings:
   - **Num Blocks**: 3-10 (3 for quick test, 10 for longer video)
   - **Resolution**: 832×480
   - **Seed**: Random or specific number
4. Click "**Start Generation**"
5. Watch video generate in real-time

### Video-to-Video Mode

1. Click "**Video-to-Video**" radio button
2. Upload a video file
3. Enter transformation prompt
4. Set **Denoising Strength**: 0.7-0.9
5. Click "**Start Generation**"

## Resolution Options

**Must be divisible by 8, minimum 64×64**

### Tested & Safe:
- **832 × 480** - Default landscape, optimized ✅
- **480 × 832** - Portrait, optimized ✅
- **512 × 512** - Square, very safe
- **640 × 480** - Faster, safe
- **768 × 432** - Good balance

### For Faster Webcam (smoother):
- **640 × 360** - Faster updates
- **512 × 288** - Very fast
- **480 × 480** - Fast square

### Higher Quality (slower):
- **1024 × 576** - Better quality
- **1280 × 720** - 720p (slower)

**Recommendation for continuous webcam**: **640×480** or **832×480**

## Settings Guide

### Denoising Strength
- **0.0-0.5**: Minimal changes (subtle style)
- **0.6-0.8**: Moderate transformation
- **0.9-0.95**: Strong transformation (recommended for webcam) ⭐
- **0.95-1.0**: Almost complete replacement

### Num Blocks
- **1-3**: Quick test
- **5-10**: Short video
- **50-100**: Longer generation
- **1000**: Continuous streaming (for webcam) ⭐

### Num Denoising Steps
- **4**: Fastest (11 fps on B200)
- **6**: Better quality (default)
- **8-12**: Highest quality (slower)

### Webcam FPS
- **10**: Balanced
- **15-20**: Smoother capture
- **30**: Maximum smoothness

## Management Commands

### Check App Status
```bash
curl https://YOUR_WORKSPACE--krea-realtime-video-serve.modal.run/health
```

### View Logs
```bash
cd /path/to/realtime-video
.venv/bin/python -m modal app logs krea-realtime-video
```

### Redeploy App
```bash
cd /path/to/realtime-video
.venv/bin/python -m modal deploy modal_app.py
```

### Check Modal Storage
```bash
.venv/bin/python -m modal volume ls krea-models
```

### Stop App
- App auto-stops after 5 minutes of inactivity
- Or stop via Modal dashboard: https://modal.com/apps/YOUR_WORKSPACE/main/deployed/krea-realtime-video

## Monitoring

### Dashboard
https://modal.com/apps/YOUR_WORKSPACE/main/deployed/krea-realtime-video

Shows:
- Container status
- Function calls
- Errors
- GPU utilization
- Cost tracking

### Browser Console
Press **F12** in browser to see:
- WebSocket connection status
- Frame rates
- Errors
- Generation logs

## Cost Tracking

**Your Modal Plan**:
- $30 free credits included
- B200: $6.25/hour when active
- Storage: ~$3/month for 107GB models
- Auto-shutdown saves costs

**Example costs**:
- 10 min testing: ~$1.04
- 1 hour session: $6.25
- Models storage: One-time download, then cached

## Troubleshooting

### Webcam transformation too subtle
- **Increase Denoising Strength to 0.9-0.95**
- Use more descriptive prompts
- Increase Num Denoising Steps to 6

### Slow/choppy generation
- Lower resolution (640×480 or 512×512)
- Reduce Num Denoising Steps to 4
- Check if B200 is actually allocated (Modal dashboard)

### WebSocket connection fails
- Check Modal logs for errors
- Refresh the page
- Check browser console (F12) for errors

### Models not loading
- Check Modal dashboard for startup errors
- Models take 1-2 minutes to load on cold start
- Verify models exist: `.venv/bin/python -m modal volume ls krea-models`

### App won't start
```bash
# View detailed logs
.venv/bin/python -m modal app logs krea-realtime-video | tail -100

# Redeploy
.venv/bin/python -m modal deploy modal_app.py
```

## Performance

**Expected on B200**:
- Generation: ~11 fps (4 inference steps)
- Latency: Very low for real-time
- VRAM Usage: ~80-120GB (B200 has 192GB)

**Optimizations in place**:
- Flash Attention 2.8.3 (Blackwell-aware)
- torch.compile enabled (DO_COMPILE=true)
- KV cache optimization
- Persistent model storage

## Tips for Best Results

### Webcam Mode
1. **Lighting**: Good lighting gives better results
2. **Background**: Simpler backgrounds transform better
3. **Prompts**: Be very descriptive and specific
4. **Denoising**: Start at 0.9, adjust up/down
5. **Blocks**: Set to 1000 for continuous operation

### Prompts
**Good webcam prompts**:
- `"anime character, Studio Ghibli style, vibrant colors, hand-drawn animation, fantasy"`
- `"cyberpunk person, intense neon lights, rain, futuristic city, dramatic blue and pink lighting"`
- `"oil painting, classical art style, renaissance, detailed brushstrokes"`
- `"pixar 3D animation, cartoon character, colorful, glossy rendering"`

**Avoid**:
- Single words: `"cyberpunk"` (too vague)
- Conflicting styles: `"realistic anime"` (contradictory)

## Technical Details

### What's Deployed
- **Image**: CUDA 12.4 devel (Ubuntu 22.04) with Python 3.11
- **PyTorch**: 2.8.0+cu128 (CUDA 12.8 runtime)
- **Flash Attention**: 2.8.3 (compiled for sm_100/Blackwell)
- **Models Path**: `/models` (Modal volume)
- **App Path**: `/root/app` (code mounted from local)
- **Config**: `configs/self_forcing_server_14b.yaml`

### Architecture
```
Client (Browser)
  ↓ WebSocket
Modal B200 Container
  ↓ Real-time Pipeline
  ├─ Webcam Frames → VAE Encoder
  ├─ Text Prompt → T5 Text Encoder
  ├─ Diffusion Model (14B params)
  └─ VAE Decoder → JPEG frames
  ↓ WebSocket
Client renders transformed video
```

### Files in Deployment
```
modal_app.py           # Deployment script
release_server.py      # FastAPI server with webcam support
templates/
  release_demo.html    # Web UI with webcam mode
configs/
  self_forcing_server_14b.yaml  # Model config
```

## Upgrading to Flash Attention 4

When FA4 becomes pip-installable (expected soon):

1. Edit `modal_app.py` line 31:
```python
.pip_install("flash-attn==4.0.0", extra_options="--no-build-isolation")
```

2. Redeploy:
```bash
.venv/bin/python -m modal deploy modal_app.py
```

FA4 will give ~20% speedup on B200 (from 11 fps → ~13 fps).

## Support

- **Modal Issues**: https://modal.com/slack
- **Model Issues**: https://github.com/krea-ai/realtime-video/issues
- **Your Modal Dashboard**: https://modal.com/apps/YOUR_WORKSPACE

## Summary

You have a **fully functional real-time webcam transformation app** running on Modal's B200 GPU with:
- ✅ PR #3 webcam support integrated
- ✅ B200 GPU with PyTorch 2.8 + Flash Attn 2.8.3
- ✅ 107GB models cached and ready
- ✅ Real-time streaming at ~11 fps
- ✅ Auto-scaling and cost optimization

**Just adjust Denoising Strength to 0.9+ for strong transformations!**
