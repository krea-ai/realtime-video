# Krea Realtime 14b
This repo contains inference code for Krea-Realtime-14b, a realtime video diffusion model distilled from Wan 2.1 14B using Self-Forcing. This repo is based on the original Self-Forcing repo, https://github.com/guandeh17/Self-Forcing .  We start our training using the LightX2V timestep distilled Wan 2.1 14B T2V checkpoint from https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill


# Self-Forcing Release Server & Sampling Quickstart

This note focuses on two entry points in the repository: the realtime websocket server in `release_server.py` (which powers `templates/release_demo.html`) and the offline sampling helpers in `sample.py`.

## Setup

create virtual environment with 
```
uv sync
```

if using flash attention, install it with 

```
uv pip install flash_attn --no-build-isolation
```

If using sageattention, you can try 
`uv pip install libs/sageattention-2.2.1-cp311-cp311-linux_x86_64.whl` or `bash install_sage.sh`

If running on B200, we recommend using flash attn 4
For H100/RTX5xxx and other GPUs, sageattention will provide the fastest speed, but it can cause oversaturated videos and faster video degradation. 

Download checkpoints from
https://huggingface.co/krea/krea-rt-14b/tree/main into `checkpoints` folder

install ffmpeg 


huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B


## Launching `release_server.py`
1. Set the environment so the server can find its assets:
   ```bash
   export MODEL_FOLDER=Wan-AI
   export CONFIG=configs/self_forcing_server_14b.yaml     # optional; defaults to this path
   export CUDA_VISIBLE_DEVICES=0                      # pick the GPU you want to serve on
   ```
2. Start the FastAPI app with Uvicorn (one worker keeps GPU memory simple):
   ```bash
   uvicorn release_server:app --host 0.0.0.0 --port 8000
   ```
   The first request loads the models defined by `CONFIG`, so expect a long warm-up on the initial run.
3. Verify the server:
   - `curl http://localhost:8000/health` should return `OK`.
   - Open `http://localhost:8000/` in a browser to reach the release demo UI (`templates/release_demo.html`), adjust the prompt/parameters, and click **Start**. Frames stream over the `/session/{uuid}` websocket.
4. Common tweaks:
   - Set `DO_COMPILE=true` to opt into `torch.compile` (longer warm-up, faster steady state).

Stop the server with `Ctrl+C`. GPU memory is released when the process exits.

## Using `sample.py`
`sample.py` reuses the same building blocks to generate clips offline without the websocket layer. It expects you to provide a fully specified `GenerateParams` instance; the helper mutates that in-place per prompt. The simplest way to sample a few prompts of your own is to create a short driver script:

```python
# sample_run.py
from pathlib import Path
from release_server import GenerateParams
from sample import sample_videos

params = GenerateParams(
    prompt="",          # placeholder; overwritten per prompt
    width=832,
    height=480,
    num_blocks=9,
    seed=42,
    kv_cache_num_frames=3,
)

prompts = [
    "A hyperrealistic close-up of ocean waves shimmering at sunset.",
    "A bustling neon-drenched alleyway with rain-soaked pavement.",
]

sample_videos(
    prompts_list=prompts,
    config_path="configs/self_forcing_dmd_will_optims.yaml",
    output_dir="outputs/samples",
    params=params,
    save_videos=True,       # requires ffmpeg
    fps=24,
)
```

Then run:
```bash
python sample_run.py
```
Key details:
- `sample_videos` loads models lazily with `load_merge_config`/`load_all` when `models=None`; reuse the returned models across calls if you sample repeatedly to avoid reload costs.
- Frames are written under the chosen `output_dir` inside subfolders named `prompt_XXX`. When `save_videos=True`, each prompt also yields an mp4 file in that directory.
- `create_grid` and `sample_single_video` near the bottom of `sample.py` offer additional helpers (grids require multiple runs to exist in `outputs/`).

If you prefer to run `sample.py` directly, edit the `if __name__ == "__main__":` block to construct a `GenerateParams` object (as in the snippet above) before calling `sample_videos`.
