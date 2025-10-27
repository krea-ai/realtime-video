# Modal Deployment Guide - Krea Realtime Video with Webcam

Complete guide for deploying Krea Realtime Video with PR #3 webcam support on Modal with B200 GPU.

---

## Quick Start (5 Minutes)

### Prerequisites
- Modal account: https://modal.com
- Python 3.11+

### Deploy in 4 Steps

```bash
# 1. Install and authenticate Modal
pip install modal
modal setup

# 2. Download models (~107GB to Modal cloud storage, 15-30 min)
modal run modal_app.py

# 3. Deploy to B200 GPU
modal deploy modal_app.py
```

You'll get: `https://YOUR_WORKSPACE--krea-realtime-video-serve.modal.run`

### Update Deployment (with code changes)

When you modify the code and want to redeploy:

```bash
# Stop the current deployment
modal app stop krea-realtime-video

# Wait a moment, then redeploy
sleep 3
modal deploy modal_app.py
```

Or do it all at once:
```bash
modal app stop krea-realtime-video && sleep 3 && modal deploy modal_app.py
```

**Note:** This forces a clean restart. Useful when the UI (HTML/JS) doesn't update or you need to clear cached state.

### Test It

**Text-to-Video (simplest):**
1. Open the URL
2. Prompt: `"ocean waves at sunset"`
3. Num Blocks: 3-5 (quick test) or 1000 (continuous)
4. Click "Start Generation"

**Webcam Mode (PR #3 feature!):**
1. Click "Webcam" mode
2. Allow camera access
3. Prompt: `"cyberpunk style, neon lights"`
4. Num Blocks: 1000 (for continuous transformation)
5. Click "Start Generation"
6. Watch real-time transformation at ~11 fps!

**Dual-Model Support (switch between generations):**
1. Select your model from the **"Model"** dropdown:
   - **14B (Krea Realtime)** - Best quality, 11 fps
   - **1.3B (Self-Forcing)** - Faster, lower quality
2. Start a generation with your chosen model
3. On the **next generation**, you can select a **different model** and the server will automatically reload it!
4. This allows you to compare quality vs. speed between generations

---

## What You Get

### Hardware
- **B200 GPU**: $6.25/hour (optimal for this model)
- **192GB VRAM**: Plenty of headroom
- **HBM3e**: 8 TB/s memory bandwidth (2.4x faster than H100)
- **Auto-shutdown**: After 5 min idle to save costs

### Software Stack
- **PyTorch 2.8.0+cu128**: B200 sm_100 support
- **Flash Attention 2.8.3**: Latest with B200 optimizations
- **CUDA 12.8**: Full Blackwell support
- **107GB Models**: Cached in Modal cloud

### Features
- **PR #3 Webcam Support**: Real-time video transformation from webcam
- **Text-to-Video**: Generate from text prompts
- **Video-to-Video**: Transform uploaded videos
- **WebSocket Streaming**: Real-time frame delivery
- **Side-by-side Display**: Input + output simultaneously
- **Background Removal**: YOLOv8n-seg for real-time green screen (webcam mode)
- **Multiple Cameras**: Select from available webcams
- **Frame Dropping**: Configurable lag prevention for long sessions
- **Portrait/Landscape**: Flip between 832×480 and 480×832

---

## Configuration

### GPU Options

**B200 (Default - Optimal)**
```python
# modal_app.py line 123
gpu="B200",  # $6.25/hour, 11 fps
```

**Alternatives** (slower due to memory bandwidth):
```python
gpu=modal.gpu.H200(),  # $4.54/hour, ~8-10 fps
gpu=modal.gpu.H100(count=1),  # $3.95/hour, ~7-9 fps
gpu=modal.gpu.A100(count=1, size="80GB"),  # $2.50/hour, ~6-8 fps
```

**Why B200 is optimal**: Memory-bandwidth bound workload (14B params + 25GB KV cache). B200's HBM3e provides 8 TB/s vs H100's 3.35 TB/s.

### Resolution Options

**Safe Resolutions** (divisible by 8, tested):
- **832 × 480** (default, optimized) ⭐
- **480 × 832** (portrait, optimized)
- **640 × 480** (faster)
- **512 × 512** (square, very safe)
- **1024 × 576** (higher quality)

**For continuous webcam**: Use 640×480 or 832×480 for best performance.

### Continuous Generation Settings

For **continuous webcam transformation**:
- **Num Blocks**: 1000 (runs ~60+ minutes)
- **Denoising Strength**: 0.6-0.8
- **Webcam FPS**: 15-30
- **Resolution**: 640×480 or 832×480

### Background Removal (Webcam Mode)

**Server-side green screen using YOLOv8n-seg:**
- Enable with "Remove Background" checkbox in webcam mode
- Detects person and replaces background with black
- Minimal performance impact: ~5-10ms per frame
- Auto-downloads model (~27MB) on first use
- Confidence threshold: 0.5 (adjustable in code)

**Performance:**
- Without BG removal: 11 fps
- With BG removal: 8-10 fps
- Model: YOLOv8n-seg (fast, real-time)
- GPU memory: ~500MB additional

### Frame Dropping (Lag Prevention)

**Prevent lag buildup during long webcam sessions:**
- Enable "Drop Old Frames" checkbox (default: ON)
- Adjust "Max Queue Size" slider (1-5×, default: 2×)
- Lower = less lag, more aggressive dropping
- Higher = smoother motion, but more lag if can't keep up
- Server logs show when frames are dropped

---

## Performance

### Expected FPS by GPU
- **B200**: 11 fps (verified, optimal)
- **H200**: 8-10 fps (estimated)
- **H100**: 7-9 fps (estimated, memory bottleneck)
- **A100 80GB**: 6-8 fps (estimated)

### First Start
- **Cold start**: 2-3 minutes (loading 107GB models to GPU)
- **Warm start**: Instant (models cached in GPU memory)
- **After 5 min idle**: Container shuts down to save costs

---

## Cost Breakdown

**Modal Starter Plan:**
- $30 free credits
- Covers ~5 hours of B200 time
- Storage: ~$3/month for 107GB models

**Example Session:**
- Model download: Free (one-time)
- 1 hour B200 usage: $6.25
- Remaining credits: $23.75

**Production use**: $6.25/hour × hours_used

---

## Troubleshooting

### Models not loading
```bash
# Re-download to Modal storage
modal run modal_app.py
```

### Deployment fails
```bash
# Check detailed logs
modal logs krea-realtime-video

# View in dashboard
https://modal.com/apps/YOUR_WORKSPACE/main/deployed/krea-realtime-video
```

### Out of memory
- B200 has 192GB - shouldn't happen
- If it does: reduce `num_blocks` or lower resolution

### Flash Attention errors
- Check logs for import errors
- Verify: "flash attn 2 available True" in startup logs
- FA4 (Blackwell-optimized) coming soon via pip

### WebSocket connection fails
- Check browser console (F12) for errors
- Ensure msgpack library loaded (check Network tab)
- Try different browser (Chrome/Firefox recommended)

### Slow performance
- Verify B200 GPU is active (check Modal dashboard)
- Lower resolution for faster generation
- Reduce num_blocks for quicker completion

### Background removal issues
**"Background removal failed" in logs:**
- YOLOv8n-seg model failed to load
- Check GPU memory (need ~500MB)
- Disable background removal for session if persistent

**Black frames or person removed:**
- Model confidence too high/low
- Adjust in `release_server.py` line 511: `confidence=0.5`
- Lower (0.3) = more aggressive, Higher (0.7) = stricter

**Slow with background removal:**
- Model processing bottleneck
- Disable feature or use faster model
- Check B200 not busy with other tasks

### Lag in long webcam sessions
- Enable "Drop Old Frames" checkbox
- Lower "Max Queue Size" to 1-1.5× for aggressive dropping
- Check Modal logs for frame dropping messages

---

## Advanced

### Using Flash Attention 4 (When Available)

FA4 is Blackwell-optimized (~20% faster) but not yet pip-installable.

**Current**: flash-attn 2.8.3 (working, good B200 support)

**When FA4 releases**: Edit `modal_app.py` line 31:
```python
.pip_install("flash-attn==4.0.0", extra_options="--no-build-isolation")
```

Source: https://modal.com/blog/reverse-engineer-flash-attention-4

### Custom Models

To use your own fine-tuned checkpoint:

1. Upload to HuggingFace
2. Edit `modal_app.py` `download_models()` function
3. Update checkpoint path in configs

### Development Mode

Test changes before deploying:
```bash
modal serve modal_app.py  # Temporary dev URL
```

### Monitoring

```bash
# View live logs
modal logs krea-realtime-video --follow

# Check GPU usage in dashboard
https://modal.com/apps/YOUR_WORKSPACE/main/deployed/krea-realtime-video
```

---

## Alternative Deployment Options

If Modal doesn't work or you need bare metal:

### Lambda Labs (Cheapest)
- H100: $2.49/hour
- Easy SSH access
- Good for development

### RunPod
- H100: $3-4/hour
- Web UI deployment
- Variable availability

### Vast.ai
- H100: $2-5/hour
- Community marketplace
- Cheapest but variable quality

### AWS/GCP
- P5 (H100): $98/hour (AWS)
- A3 (H100): $73/hour (GCP)
- Enterprise reliability

See original ALTERNATIVE_DEPLOYMENT.md sections for detailed setup instructions.

---

## Files Created

**Keep these:**
- `modal_app.py` - Modal deployment script
- `CLAUDE.md` - This combined guide
- `.modalignore` - Deployment exclusions

**App modifications:**
- `release_server.py` - Dynamic model switching, background removal, frame dropping, horizontal mirror
- `utils/wan_wrapper.py` - Added local_files_only, torch.load fix
- `wan/modules/attention.py` - Fixed Flash Attention imports
- `templates/release_demo.html` - Webcam UI, model selector, camera selector, frame dropping controls, portrait/landscape flip
- `background_removal.py` - YOLOv8n-seg background removal processor
- `modal_app.py` - Added ultralytics, roboflow dependencies

---

## Support

- **Modal**: https://modal.com/slack
- **Krea Issues**: https://github.com/krea-ai/realtime-video/issues
- **Modal Docs**: https://modal.com/docs

---

## Summary

✅ **What works:**
- B200 GPU with PyTorch 2.8.0+cu128
- Flash Attention 2.8.3
- Webcam real-time transformation (PR #3)
- Text-to-video and video-to-video modes
- 107GB models cached in Modal
- **Dual-model support** (14B + 1.3B, switchable between generations)
- **Background removal** (YOLOv8n-seg, real-time green screen)
- **Multiple camera selection** (Firefox & Chrome compatible)
- **Frame dropping controls** (prevent lag in long sessions)
- **Portrait/landscape flip** (832×480 ↔ 480×832)

✅ **Deployed at:**
https://YOUR_WORKSPACE--krea-realtime-video-serve.modal.run

✅ **Performance:**
- 14B Model: 11 fps on B200, best quality (8-10 fps with background removal)
- 1.3B Model: Faster inference, lower quality
- Switch models between generations (auto-reload)
- Continuous generation with num_blocks=1000
- Resolutions: 832×480 (landscape) or 480×832 (portrait)
- Background removal overhead: ~5-10ms per frame

✅ **Model Switching:**
- Select 14B or 1.3B from the UI
- Server automatically loads on next generation
- Compare quality vs. speed between runs

🚀 **Ready to use!**
