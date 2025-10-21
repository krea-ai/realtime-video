import torch
torch.set_grad_enabled(False)
from safetensors.torch import load_file, save_file
import safetensors
from functools import lru_cache
from collections import deque
from utils.scheduler import SchedulerInterface, FlowMatchScheduler

import asyncio
import random
import os
import random
import gc
import logging
import base64
import threading
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, Dict, List
import traceback
from typing import Callable, TYPE_CHECKING
import queue
import uuid
import socket
import tempfile
import shutil
import subprocess
import numpy as np

if TYPE_CHECKING:
    from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
    from demo_utils.vae_block3 import VAEEncoderWrapper, VAEDecoderWrapper
    from pipeline import CausalInferencePipeline

# Internal / self-forcing imports
from v2v import encode_video_latent, get_denoising_schedule
from utils.misc import AtomicCounter

# External imports
from omegaconf import OmegaConf
from tqdm import tqdm
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ValidationError
import time
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from msgpack import packb, unpackb


from settings import MODEL_FOLDER
from wan.modules.vae import WanVAE
import torch._dynamo as dynamo
dynamo.config.recompile_limit = 32

# Helper function for resampling frames
def resample_array(array, target_length):
    """Resample a list to the target length using linear interpolation of indices"""
    if len(array) == target_length:
        return array
    indices = np.round(np.linspace(0, len(array) - 1, target_length)).astype(int)
    return [array[i] for i in indices]

# SECTION: CONFIGURATION AND CONSTANTS

# Basic configuration - logs to console by default
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Get a logger
log = logging.getLogger(__name__)

# Global storage for session frames
session_frames_storage: Dict[str, List[torch.Tensor]] = {}
session_frame_locks: Dict[str, threading.Lock] = {}

UUID_NIL = str(uuid.UUID(int=0))
USE_STATIC_ENCODER_COND_DICT = os.getenv("USE_STATIC_ENCODER_COND_DICT", "false").lower() in ("true", "1", "yes")

DO_COMPILE = os.getenv("DO_COMPILE", "false").lower() in ("true", "1", "yes")
print("DO_COMPILE", DO_COMPILE)

gpu = torch.cuda.current_device()
upload_stream = torch.cuda.Stream(device=gpu)
download_stream = torch.cuda.Stream(device=gpu)

def load_merge_config(config_path: str | Path) -> OmegaConf:
    config = OmegaConf.load(config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    merged_config = OmegaConf.merge(
        default_config, config
    )
    return merged_config

class Models:
    """
    Wrapper class that holds all loaded models
    """
    def __init__(self, text_encoder, transformer, pipeline, vae_encoder, vae_decoder):
        self.text_encoder: WanTextEncoder = text_encoder
        self.transformer: WanDiffusionWrapper = transformer
        self.pipeline: CausalInferencePipeline = pipeline
        self.vae_encoder: VAEEncoderWrapper = vae_encoder
        self.vae_decoder: VAEDecoderWrapper = vae_decoder

def copy_models(models: Models, config, gpu):
    from copy import deepcopy
    with torch.cuda.device(gpu):
        text_encoder = deepcopy(models.text_encoder).to(gpu)
        transformer = deepcopy(models.transformer).to(gpu)
        vae_encoder = deepcopy(models.vae_encoder).to(gpu)
        vae_decoder = deepcopy(models.vae_decoder).to(gpu)
        pipeline = load_pipeline(config, gpu, transformer, text_encoder, vae_decoder)
        return Models(text_encoder, transformer, pipeline, vae_encoder, vae_decoder)

def load_text_encoder():
    """Load and configure the text encoder model"""
    t_start = time.time()

    if USE_STATIC_ENCODER_COND_DICT:
        # temp code to just return a static embedding
        # TODO: remove
        print("USING STATIC COND DICT. PLSPLSPLS REMOVE BEFORE MERGING")
        static_cond_dict = torch.load("static_cond_dict_cat_skateboard.pth")
        class StaticTextEncoder(torch.nn.Module):
            def forward(self, text_prompts):
                return static_cond_dict
        return StaticTextEncoder()
    
    from utils.wan_wrapper import WanTextEncoder
    t_import = time.time()
    log.debug(f"Text encoder import took: {t_import - t_start:.2f}s")
    
    text_encoder = WanTextEncoder()
    text_encoder.eval()
    text_encoder.to(dtype=torch.bfloat16)
    text_encoder.requires_grad_(False)
    
    t_finish = time.time()
    log.debug(f"Text encoder load completed in: {t_finish - t_import:.2f}s, total: {t_finish - t_start:.2f}s")
    
    return text_encoder


def load_transformer(config, meta_transformer=False):
    """Load and configure the transformer model"""
    t_start = time.time()

    checkpoint_path = config.checkpoint_path
    
    from utils.wan_wrapper import WanDiffusionWrapper
    t_import = time.time()
    log.debug(f"Transformer import took: {t_import - t_start:.2f}s")
    
    state_dict = load_file(checkpoint_path, device="cuda")
    log.debug(f"Loading transformer state dict from {checkpoint_path}")
    if state_dict["model.blocks.0.self_attn.k.weight"].shape[0] == 1536:
        model_name = "Wan2.1-T2V-1.3B"
    else:
        model_name = "Wan2.1-T2V-14B"

    timestep_shift = getattr(config, "timestep_shift", 5.0)
    transformer = WanDiffusionWrapper(model_name=model_name, timestep_shift=timestep_shift, is_causal=True)
    transformer.load_state_dict(state_dict)
    
    transformer = transformer.to(dtype=torch.bfloat16)
    transformer.eval()
    transformer.requires_grad_(False)
    transformer.to(torch.cuda.current_device())

    for block in transformer.model.blocks:
        block.self_attn.fuse_projections()

    if config.enable_fp8:
        log.debug("Quantizing transfofmer to fp8")
        from torchao.quantization.quant_api import quantize_, Float8DynamicActivationFloat8WeightConfig, PerTensor
        quantize_(transformer, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))

    t_finish = time.time()
    log.debug(f"Transformer load completed in: {t_finish - t_import:.2f}s, total: {t_finish - t_start:.2f}s")

    
    return transformer

def load_vae():
    """Load and configure the VAE decoder"""
    t_start = time.time()

    log.debug("Using demo_utils.vae_block3.VAEEncoderWrapper")
    log.debug("Using demo_utils.vae_block3.VAEDecoderWrapper")
    from demo_utils.vae_block3 import VAEEncoderWrapper
    from demo_utils.vae_block3 import VAEDecoderWrapper
    vae_dtype = torch.float16
    vae_path = os.path.join(MODEL_FOLDER, "Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth")
    vae = WanVAE(vae_pth=vae_path, dtype=vae_dtype)
    vae_encoder = VAEEncoderWrapper(vae)

    vae_decoder = VAEDecoderWrapper()
    vae_state_dict = torch.load(vae_path, map_location="cpu")
    decoder_state_dict = {}
    for key, value in vae_state_dict.items():
        if 'decoder.' in key or 'conv2' in key:
            decoder_state_dict[key] = value

    vae_encoder.eval()
    vae_encoder.to(dtype=torch.float16)
    vae_encoder.requires_grad_(False)
    vae_encoder.to(torch.cuda.current_device())

    keys = vae_decoder.load_state_dict(decoder_state_dict, strict=False)
    print(f"Incompatible {keys} while loading vae decoder")
    vae_decoder.eval()
    vae_decoder.to(dtype=torch.float16)
    vae_decoder.requires_grad_(False)
    vae_decoder.to(torch.cuda.current_device())

    t_finish = time.time()
    log.debug(f"VAE load completed in: {t_finish - t_start:.2f}s")
    
    return vae_encoder, vae_decoder


def load_pipeline(config, device, transformer, text_encoder, vae_decoder):
    """Initialize the causal inference pipeline"""
    t_start = time.time()
    
    from pipeline import CausalInferencePipeline
    t_import = time.time()
    log.debug(f"Pipeline import took: {t_import - t_start:.2f}s")
    
    pipeline = CausalInferencePipeline(
        config,
        device=device,
        generator=transformer,
        text_encoder=text_encoder,
        vae=vae_decoder
    )

    
    t_finish = time.time()
    log.debug(f"Pipeline initialization completed in: {t_finish - t_start:.2f}s")
    
    return pipeline

def load_all(config: OmegaConf, meta_transformer=False):
    """
    Load all models with progress tracking
    
    Args:
        config: OmegaConf configuration object containing checkpoint_path and use_trt settings
    
    Returns:
        Models instance containing all loaded models
    """
    log.info("Starting model loading...")
    t_total_start = time.time()
    
    # Extract settings from config
    checkpoint_path = config.get('checkpoint_path', './checkpoints/self_forcing_dmd.pt')
    log.debug(f"Using checkpoint: {checkpoint_path}")
    
    
    # Create progress bar with 4 stages
    with tqdm(total=4, desc="Loading models") as pbar:
        # Load transformer
        pbar.set_description("Loading transformer")
        t_stage_start = time.time()
        transformer = load_transformer(config)
        log.debug(f"Loading transformer took: {time.time() - t_stage_start:.2f}s")
        pbar.update(1)

        # Load text encoder
        pbar.set_description("Loading text encoder")
        t_stage_start = time.time()
        text_encoder = load_text_encoder()
        log.debug(f"Loading text encoder took: {time.time() - t_stage_start:.2f}s")
        pbar.update(1)
        gc.collect()
        torch.cuda.empty_cache()
        
        # Load VAE decoder
        pbar.set_description("Loading VAE")
        t_stage_start = time.time()
        vae_encoder, vae_decoder = load_vae()
        log.debug(f"Loading VAE took: {time.time() - t_stage_start:.2f}s")
        pbar.update(1)

        # Initialize pipeline
        pbar.set_description("Initializing pipeline")
        t_stage_start = time.time()
        pipeline = load_pipeline(config, torch.cuda.current_device(), transformer, text_encoder, vae_decoder)
        log.debug(f"Initializing pipeline took: {time.time() - t_stage_start:.2f}s")
        pbar.update(1)
    
    t_total_end = time.time()
    log.info(f"All models loaded successfully in {t_total_end - t_total_start:.2f}s")
    
    models = Models(text_encoder, transformer, pipeline, vae_encoder, vae_decoder)

    gc.collect()
    torch.cuda.empty_cache()
    if DO_COMPILE:
        print("compiling models")
        compile_models(models)
    gc.collect()
    torch.cuda.empty_cache()

    return models

class GenerateParams(BaseModel):
    prompt: str
    width: int = 832
    height: int = 480
    
    seed: int | None = None
    resume_latents: bytes | None = None
    strength: float = 1.0
    request_id: str | None = None

    interp_blocks: int = -1
    context_noise: float = 0.0
    keep_first_frame: bool = False
    kv_cache_num_frames: int = 3
    num_blocks: int = 9
    num_denoising_steps: int | None = 5 # use 4 for performance

    block_on_frame: bool = False

    input_video: str | None = None
    start_frame: bytes | str | None = None
    timestep_shift: float = 5.0

    webcam_mode: bool = False
    webcam_fps: int = 10
    class Config:
        arbitrary_types_allowed = True
    

class GenerationSession:
    SESSION_COUNTER = AtomicCounter()

    @torch.inference_mode()
    def __init__(self, params: GenerateParams,
                 config: OmegaConf, debug=False, frame_callback: Optional[Callable] = None, models: Models | None = None):
        self.current_use_taehv = config.use_taehv
        self.frame_callback = frame_callback or (lambda *args, **kwargs: logging.warning("No frame callback set!"))
        self.session_id = self.SESSION_COUNTER.increment()

        self.frame_queue: queue.Queue[torch.Tensor] = queue.Queue()
        self.block_idx = 0
        self.params = params


        self.input_video = params.input_video
        if self.input_video is None and not params.webcam_mode:
            self.params.strength = 1.0
            
        self.start_frame = params.start_frame

        self.width = params.width // 8 * 8
        self.height = params.height // 8 * 8 
        self.latent_width = self.width // 8
        self.latent_height = self.height // 8
        self.resume_latents: Optional[torch.Tensor] = None
        self.last_frame_latent = None
        
        self.config = config
        self.debug = debug
        self.gpu = torch.cuda.current_device()

        # Generation params pulled from prev global fars
        self.interpolated_prompt_embeds = []
        self.original_prompt_embeds = None

        # We'll store the current prompt embeds at each block
        self.current_prompt_embeds = None

        # Initialze with passed vaplue
        self.context_noise = params.context_noise
        self.kv_cache_num_frames = params.kv_cache_num_frames
        self.g_num_blocks = self.num_blocks = params.num_blocks

        frame_cache_len = 1 + (params.kv_cache_num_frames - 1) * 4 
        self.frame_context_cache = deque(maxlen=frame_cache_len)

        
        self.gpu_initialized = False

        self.encode_vae_cache: list[Optional[torch.Tensor]] = [None] * 55
        self.decode_vae_cache: list[Optional[torch.Tensor]] = [None] * 55
        self.num_frame_per_block = 3
       
        self.rnd = torch.Generator(self.gpu).manual_seed(self.params.seed)

        num_latent_frames = self.num_blocks * self.num_frame_per_block

        latent_shape = [1, num_latent_frames, 16, self.latent_height, self.latent_width]
        self.all_latents = torch.zeros(latent_shape, device=self.gpu, dtype=torch.bfloat16).contiguous()
        self.noise = torch.randn(latent_shape, device=self.gpu, dtype=torch.bfloat16, generator=self.rnd).contiguous()

        # Generation parameters
        self.current_start_frame = 0
        self.total_frames_sent = 0

        self.disposed = threading.Event()
        self.generation_lock = asyncio.Lock()

        self.init_models(models, self.params)
        self.models = models

        self.denoising_step_list = get_denoising_schedule(
            self.zero_padded_timesteps, self.params.strength, steps=self.params.num_denoising_steps
        )

        print("denoising step list: ", self.denoising_step_list)
        if self.input_video is not None:
            init_denoising_strength_scaled = self.denoising_step_list[0] / 1000
            latents, _ = self.encode_v2v(self.input_video, max_frames=None, resample_to=None)
            latents = latents[None].to(self.gpu, dtype=self.noise.dtype).movedim(1, 2)

            self.noise = latents * (1.0 - init_denoising_strength_scaled) + torch.randn(latents.shape, device=self.noise.device, dtype=self.noise.dtype, generator=self.rnd) * init_denoising_strength_scaled
            self.noise = self.noise.contiguous()
            print("latents shape: ", latents.shape)
            actual_num_blocks = latents.shape[1] // self.num_frame_per_block - 1
            self.num_blocks = min(actual_num_blocks, self.params.num_blocks)
            print("final num blocks: ", self.num_blocks)
        if self.params.start_frame is not None:
            print("Setting up start frame")
            self.setup_start_frame(self.params.start_frame, models)

        self.last_pred: Optional[torch.Tensor] = None
    
    def to(self, gpu):
        if self.gpu == gpu: return
        self.all_latents = self.all_latents.to(gpu, non_blocking=True)
        self.noise = self.noise.to(gpu, non_blocking=True)

        if isinstance(self.encode_vae_cache, list):
            self.encode_vae_cache = [cache.to(gpu, non_blocking=True) if cache is not None else None for cache in self.encode_vae_cache]
        if isinstance(self.decode_vae_cache, list):
            self.decode_vae_cache = [cache.to(gpu, non_blocking=True) if cache is not None else None for cache in self.decode_vae_cache]
        
        # move prompt embed stuff
        self.current_prompt_embeds = self.current_prompt_embeds.to(gpu, non_blocking=True) if self.current_prompt_embeds is not None else None
        if hasattr(self, 'conditional_dict'):
            for key, value in self.conditional_dict.items():
                self.conditional_dict[key] = value.to(gpu, non_blocking=True)
        self.interpolated_prompt_embeds = [embed.to(gpu, non_blocking=True) for embed in self.interpolated_prompt_embeds]
        self.gpu = gpu
    
    def dispose(self):
        self.disposed.set()

    def interpolate_prompt_embeds(self, models: Models, new_prompt, interpolation_steps):
        if self.current_prompt_embeds is None: return
        prompt_embeds_1 = self.current_prompt_embeds
        prompt_embeds_2 = models.text_encoder(text_prompts=[new_prompt])["prompt_embeds"].to(dtype=torch.bfloat16)
        x = torch.lerp(
            prompt_embeds_1, 
            prompt_embeds_2, 
            torch.linspace(0, 1, steps=interpolation_steps).unsqueeze(1).unsqueeze(2).to(prompt_embeds_1)
        )
        self.interpolated_prompt_embeds = list(x.chunk(interpolation_steps, dim=0))

    def push_frame(self, frame: str | bytes, denoising_strength: float | None = None, request_id: str | None = None):
        try:
            if denoising_strength is not None:
                self.params.strength = denoising_strength
            if isinstance(frame, str):
                if frame.startswith("data:"):
                    frame = frame[frame.index(",") + 1:]
                frame = base64.b64decode(frame)
            image = Image.open(BytesIO(frame)).convert("RGB")
            tensor = TF.to_tensor(image).to(dtype=torch.float16).pin_memory()
            with torch.cuda.stream(upload_stream):
                tensor = tensor.to(self.gpu).sub_(0.5).mul_(2.0)
            tensor.request_id = request_id
            self.frame_queue.put(tensor)
        except Exception as e:
            traceback.print_exc()
            print(f"Killing from push_frame: {e}")
            self.dispose()
    
    def process_webcam_frames(self, models: Models, idx: int):
        """Process webcam frames for streaming v2v with proper frame encoding"""
        # Determine number of frames to encode based on block index
        if idx == 0:
            num_frames_to_encode = 9
        else:
            num_frames_to_encode = 12

        # Wait until we have at least enough frames
        while self.frame_queue.qsize() < num_frames_to_encode:
            if self.disposed.is_set():
                return None
            time.sleep(0.01)  # Check every 10ms to avoid busy-spinning

        frame_list = []
        while not self.frame_queue.empty():
            try:
                frame_list.append(self.frame_queue.get_nowait())
            except queue.Empty:
                break

        if len(frame_list) < num_frames_to_encode:
            return None

        # Resample to target number of frames for temporal spacing
        frames_to_encode = resample_array(frame_list, num_frames_to_encode)

        frames_tensor = torch.stack(frames_to_encode)

        latents, self.encode_vae_cache = encode_video_latent(
            models.vae_encoder,
            self.encode_vae_cache,
            frames=frames_tensor,
            height=self.params.height,
            width=self.params.width,
            stream=idx > 0,
        )

        return latents

    @lru_cache(maxsize=32)
    def encode_v2v(self, video_path_or_url: str, max_frames=None, resample_to=None):
        latents = encode_video_latent(self.models.vae_encoder,
                                    encode_vae_cache=[None] * 55,
                                    video_path_or_url=video_path_or_url,
                                    height=self.params.height,
                                    width=self.params.width,
                                    stream=False,
                                    max_frames=max_frames,
                                    resample_to=resample_to,
                                    )
        return latents
        
    def init_models(self, models: Models, params: GenerateParams):
        attn_size = self.params.kv_cache_num_frames + models.pipeline.num_frame_per_block
        for block in models.pipeline.generator.model.blocks:
            block.self_attn.local_attn_size = -1
        models.pipeline.local_attn_size = attn_size
        for block in models.pipeline.generator.model.blocks:
            block.self_attn.local_attn_size = -1
        models.pipeline._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=gpu)
        models.pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=gpu)
        models.pipeline.generator.model.block_mask = None 



        # this prevents a cuda sync
        models.pipeline.scheduler = FlowMatchScheduler(shift=params.timestep_shift, sigma_min=0.0, extra_one_step=True)
        models.pipeline.scheduler.set_timesteps(1000, training=True)
        
        st = models.pipeline.scheduler.timesteps
        self.zero_padded_timesteps = torch.cat((st.cpu(), torch.tensor([0], dtype=torch.float32))).to(torch.cuda.current_device())


    def get_clean_context_frames(self, models: Models):
        current_kv_cache_num_frames = self.kv_cache_num_frames if self.kv_cache_num_frames is not None else self.params.kv_cache_num_frames
        clean_context_frames = self.all_latents[:, :self.current_start_frame]
        if self.params.keep_first_frame or (self.block_idx - 1) * models.pipeline.num_frame_per_block < current_kv_cache_num_frames:
            if current_kv_cache_num_frames == 1:
                clean_context_frames = clean_context_frames[:, :1]
            else:
                clean_context_frames = torch.cat((clean_context_frames[:, :1], clean_context_frames[:,1:][:, -current_kv_cache_num_frames + 1:]), dim=1)
        else:
            # print("reencoding first latent frame, block idx:")
            clean_context_frames = clean_context_frames[:,1:][:, -current_kv_cache_num_frames + 1:]
            first_frame_latent = encode_video_latent(models.vae_encoder, [None]*55, resample_to=16, max_frames=81, video_path_or_url=None, frames=self.frame_context_cache[0][0].half(), height=480, width=832, stream=False,)[0].transpose(0, 1)[None]
            clean_context_frames = torch.cat((first_frame_latent, clean_context_frames), dim=1).to(self.all_latents)
        return clean_context_frames
    
    def setup_start_frame(self, image: Image.Image, models: Models):
        num_context_frames = self.params.kv_cache_num_frames
        frame_cache_len = 1 + (num_context_frames - 1) * 4 

        tensor = TF.to_tensor(image).to(dtype=torch.float16)
        tensor = tensor.to("cuda").sub_(0.5).mul_(2.0)
        tensors = torch.stack([tensor] * frame_cache_len)
        latents = encode_video_latent(models.vae_encoder, [None]*55, resample_to=16, max_frames=81, video_path_or_url=None, frames=tensors, height=480, width=832, stream=False)[0].transpose(0, 1)[None]
        self.resume_latents = latents

    def recompute_kv_cache(self, models: Models):
        if self.block_idx == 0:
            models.pipeline._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=self.gpu)
            if self.resume_latents is not None:
                print("Resuming generation from latents, shape", self.resume_latents.shape)
                self.current_start_frame = self.resume_latents.shape[1]
                self.all_latents[:, :self.current_start_frame] = self.resume_latents
            else:
                return self.current_start_frame

        for block in models.pipeline.generator.model.blocks:
            block.self_attn.num_frame_per_block = models.pipeline.num_frame_per_block

        current_kv_cache_num_frames = self.params.kv_cache_num_frames
        
        model_input_start_frame = min(self.current_start_frame, current_kv_cache_num_frames)
                
        clean_context_frames = self.get_clean_context_frames(models)
        
        models.pipeline._initialize_kv_cache(
            batch_size=clean_context_frames.shape[0], dtype=clean_context_frames.dtype, device=clean_context_frames.device
        )

        block_mask = models.pipeline.generator.model._prepare_blockwise_causal_attn_mask(
            device=str(clean_context_frames.device),
            num_frames=clean_context_frames.shape[1],
            frame_seqlen=models.pipeline.frame_seq_length,
            num_frame_per_block=models.pipeline.num_frame_per_block,
            local_attn_size=-1,
        )

        context_timestep = torch.ones(
            [clean_context_frames.shape[0], clean_context_frames.shape[1]],
            device=clean_context_frames.device,
            dtype=torch.int64) * 0
        models.pipeline.generator.model.block_mask = block_mask
        models.transformer(
            noisy_image_or_video=clean_context_frames,
            conditional_dict=self.conditional_dict,
            timestep=context_timestep,
            kv_cache=models.pipeline.kv_cache1,
            crossattn_cache=models.pipeline.crossattn_cache,
            current_start=model_input_start_frame * models.pipeline.frame_seq_length,
        )
        models.pipeline.generator.model.block_mask = None
        return model_input_start_frame

    @torch.inference_mode()
    def generate_block_internal(self, models: Models):
        idx = self.block_idx
        if idx >= self.num_blocks:
            return None

        if self.current_prompt_embeds is None:
            self.conditional_dict = models.text_encoder(text_prompts=[self.params.prompt])
            for key, value in self.conditional_dict.items():
                self.conditional_dict[key] = value.to(dtype=torch.bfloat16).contiguous()
            self.current_prompt_embeds = self.conditional_dict['prompt_embeds']

        model_input_start_frame = self.recompute_kv_cache(models)
        assert model_input_start_frame is not None
        frame_ids: list[str | None] = []

        if self.params.webcam_mode:
            latents = self.process_webcam_frames(models, idx)
            if latents is None:
                return None

            denoising_strength_scaled = self.denoising_step_list[0] / 1000.0
            latents = latents[None].to("cuda", dtype=self.noise.dtype).movedim(1, 2)
            noisy_input = latents * (1.0 - denoising_strength_scaled) + torch.randn_like(latents) * denoising_strength_scaled
        else:
            noisy_input = self.noise[:, self.current_start_frame:self.current_start_frame + models.pipeline.num_frame_per_block]

        if self.interpolated_prompt_embeds:
            models.pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=self.gpu)
            next_interpolated_text_emb = self.interpolated_prompt_embeds.pop(0)
            self.current_prompt_embeds = next_interpolated_text_emb.to(
                dtype=self.current_prompt_embeds.dtype, device=self.current_prompt_embeds.device)

        # This is set from config
        for index, current_timestep in enumerate(self.denoising_step_list):
            if self.disposed.is_set(): return
            t_step_start = time.time()
            # Normal initialize timestamp stuff
            timestep = torch.ones([1, models.pipeline.num_frame_per_block], device=self.noise.device,
                                dtype=torch.int64) * current_timestep

            self.conditional_dict['prompt_embeds'] = self.current_prompt_embeds
            
            if index < len(self.denoising_step_list) - 1:
                start_time = time.time()
                _, denoised_pred = models.transformer(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=models.pipeline.kv_cache1,
                    crossattn_cache=models.pipeline.crossattn_cache,
                    current_start=model_input_start_frame * models.pipeline.frame_seq_length
                )
                start_time = time.time()
                next_timestep = self.denoising_step_list[index + 1]
                noisy_input = models.pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn(*denoised_pred.flatten(0, 1).shape, generator=self.rnd, device=denoised_pred.device, dtype=torch.bfloat16),
                    next_timestep * torch.ones([1 * models.pipeline.num_frame_per_block], device=self.noise.device, dtype=torch.long, )
                ).unflatten(0, denoised_pred.shape[:2])
                # logging.debug("(renoise) time %f", time.time() - start_time)
            else:
                start_time = time.time()
                # otherwise just denoise
                _, denoised_pred = models.transformer(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=models.pipeline.kv_cache1,
                    crossattn_cache=models.pipeline.crossattn_cache,
                    current_start=model_input_start_frame * models.pipeline.frame_seq_length
                )

        self.all_latents[:, self.current_start_frame:self.current_start_frame + models.pipeline.num_frame_per_block] = denoised_pred
        self.last_pred = denoised_pred
        decode_start = time.time()
        
        if (self.params.width, self.params.height) != (832, 480):
            print("Falling back to eager for VAE decode")
            ctx = torch.compiler.set_stance("force_eager")
        else:
            ctx = torch.compiler.set_stance("default")
        
        with ctx:
            pixels, self.decode_vae_cache = models.vae_decoder(denoised_pred.half(), *self.decode_vae_cache)
        
        self.frame_context_cache.extend(pixels.split(1, dim=1))
        if idx == 0:
            pixels = pixels[:, 3:, :, :, :]  # Skip first 3 frames of first block

        self.most_recent_frame = pixels[:, -1:, ...].clone()

        event = torch.cuda.Event()
        event.record()
        self.frame_callback(pixels, frame_ids, event)

        self.current_start_frame += models.pipeline.num_frame_per_block
        self.total_frames_sent += pixels.shape[1]
        self.block_idx += 1
        self.resume_latents = None
        
        return pixels
    
    @torch.inference_mode()
    def generate_block(self, models: Models):
        with torch.cuda.device(self.gpu):
            out = self.generate_block_internal(models)
            if out is None:
                raise asyncio.CancelledError()
            return out
    
    def generate_blocks(self, num_blocks: int, models: Models):
        for _ in range(num_blocks):
            self.generate_block(models)
    
    def __hash__(self):
        return id(self)

def compile_models(models: Models):
    models.vae_decoder = torch.compile(models.vae_decoder, fullgraph=True,) 
    models.transformer = torch.compile(models.transformer)

# SECTION - SERVER & HANDLING
async def lifespan(app: FastAPI):
    app.state.config = load_merge_config(os.getenv("CONFIG", "configs/self_forcing_server_14b.yaml"))
    app.state.models = load_all(app.state.config)
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return "OK"

@app.get("/")
async def root():
    demo_path = Path(__file__).parent / "templates" / "release_demo.html"
    if not demo_path.exists():
        log.warning("Release demo template missing at %s", demo_path)
        return HTMLResponse("<h1>Self-Forcing</h1><p>Demo UI not found.</p>", status_code=404)
    return HTMLResponse(demo_path.read_text(encoding="utf-8"))

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file and return its temporary path for use in generation"""
    try:
        # Create a temporary file with the original extension
        suffix = Path(file.filename).suffix if file.filename else ".mp4"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        # Copy uploaded file to temp file
        with temp_file:
            shutil.copyfileobj(file.file, temp_file)
        
        log.info(f"Video uploaded to temporary file: {temp_file.name}")
        return JSONResponse({"path": temp_file.name, "filename": file.filename})
    except Exception as e:
        log.error(f"Error uploading video: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        file.file.close()

@app.post("/upload_start_frame")
async def upload_start_frame(file: UploadFile = File(...)):
    """Upload a start frame image and return its temporary path for use in generation"""
    try:
        # Create a temporary file with the original extension
        suffix = Path(file.filename).suffix if file.filename else ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        # Copy uploaded file to temp file
        with temp_file:
            shutil.copyfileobj(file.file, temp_file)
        
        log.info(f"Start frame uploaded to temporary file: {temp_file.name}")
        return JSONResponse({"path": temp_file.name, "filename": file.filename})
    except Exception as e:
        log.error(f"Error uploading start frame: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        file.file.close()

@app.get("/download_video/{session_id}")
async def download_video(session_id: str):
    """Download the generated video as MP4 for a given session"""
    from fastapi.responses import Response
    
    # Check if we have frames for this session
    if session_id not in session_frames_storage:
        return JSONResponse({"error": "No video data found for this session"}, status_code=404)
    
    # Get the frames for this session
    frames = session_frames_storage[session_id]
    if not frames:
        return JSONResponse({"error": "No frames available"}, status_code=404)
    
    try:
        # Combine all frame tensors
        # frames is a list of tensors, each with shape [1, num_frames, 3, H, W]
        all_frames = torch.cat(frames, dim=1)  # Shape: [1, total_frames, 3, H, W]
        
        # Save to MP4 using ffmpeg
        mp4_data = save_video_to_bytes(all_frames, fps=16)
        
        if mp4_data is None:
            return JSONResponse({"error": "Failed to generate MP4"}, status_code=500)
        
        # Clean up the stored frames
        del session_frames_storage[session_id]
        if session_id in session_frame_locks:
            del session_frame_locks[session_id]
        
        # Return the MP4 file
        return Response(
            content=mp4_data,
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename=video_{session_id}.mp4"
            }
        )
        
    except Exception as e:
        log.error(f"Error generating video: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

def save_video_to_bytes(pixels: torch.Tensor, fps: int = 24) -> Optional[bytes]:
    """Save video frames to MP4 and return as bytes"""
    try:
        # pixels shape: [1, num_frames, 3, H, W]
        video_tensor = pixels[0].cpu().clamp(0, 1)
        num_frames, _, height, width = video_tensor.shape
        
        # Convert to uint8 RGB frames
        video_np = (video_tensor.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        
        # Create a temporary file for output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_path = tmp_file.name
        
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
            tmp_path
        ]
        
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        process.stdin.write(video_np.tobytes())
        process.stdin.close()
        process.wait()
        
        if process.returncode != 0:
            log.error(f"FFmpeg error: {process.stderr.read().decode()}")
            return None
        
        # Read the MP4 file
        with open(tmp_path, 'rb') as f:
            mp4_data = f.read()
        
        # Clean up
        os.unlink(tmp_path)
        
        return mp4_data
        
    except Exception as e:
        log.error(f"Error creating video: {e}")
        return None

generate_pool = ThreadPoolExecutor(max_workers=1)
encode_pool = ThreadPoolExecutor(max_workers=24)

async def ws_session(websocket: WebSocket, id: str, config: OmegaConf, models: Models):
    loop = asyncio.get_event_loop()
    await websocket.accept()

    await websocket.send_json({"status": "ready", "worker": socket.gethostname()})

    session = None
    frame_sender_task = None
    generate_task = None

    try:
        while True:
            try:
                params = GenerateParams.model_validate(unpackb(await websocket.receive_bytes()))
                break
            except ValidationError as e:
                await websocket.send_json({"error": e.errors()})
                continue
        params.block_on_frame = True
        if params.seed is None:
            params.seed = random.randint(0, 2**24 - 1)

        # Convert start_frame path to PIL Image if provided
        if params.start_frame is not None and isinstance(params.start_frame, str):
            try:
                params.start_frame = Image.open(params.start_frame).convert("RGB")
                log.info(f"Loaded start frame image: {params.start_frame.size}")
            except Exception as e:
                log.error(f"Failed to load start frame: {e}")
                params.start_frame = None

        # Initialize session frame storage
        if id not in session_frames_storage:
            session_frames_storage[id] = []
            session_frame_locks[id] = threading.Lock()
        
        frame_queue = asyncio.Queue[asyncio.Future[bytes]]()
        async def frame_sender():
            while True:
                try:
                    next_frame = await (await frame_queue.get())
                    await websocket.send_bytes(next_frame)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logging.error(f"Error sending frame: {e}")
                frame_queue.task_done()
        frame_sender_task = asyncio.create_task(frame_sender())

        async def extract_frame(frames_future: asyncio.Future[torch.Tensor], idx: int, frame_id: str) -> bytes:
            io = BytesIO()
            frames = await frames_future
            await loop.run_in_executor(encode_pool, lambda: TF.to_pil_image(frames[0, idx], "RGB").save(io, format='JPEG', quality=90))
            if websocket.query_params.get("fmt", "jpeg") == "msgpack":
                return packb({ "image": io.getvalue(), "request_id": frame_id })
            return io.getvalue()

        def frame_callback(tensor: torch.Tensor, frame_ids: list[str], event: torch.cuda.Event):
            def get_cpu_frames():
                cpu_tensor = torch.zeros_like(tensor, device="cpu", pin_memory=True)
                download_stream.wait_event(event)
                with torch.cuda.stream(download_stream):
                    cpu_tensor.copy_(tensor)
                return cpu_tensor.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
            
            def store_frames():
                cpu_tensor = torch.zeros_like(tensor, device="cpu", pin_memory=True)
                download_stream.wait_event(event)
                with torch.cuda.stream(download_stream):
                    cpu_tensor.copy_(tensor)
                normalized = cpu_tensor.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
                with session_frame_locks[id]:
                    session_frames_storage[id].append(normalized.clone())
                return normalized

            try:
                # Store frames and also send them
                cpu_frame_future = loop.run_in_executor(encode_pool, store_frames)

                for idx in range(tensor.shape[1]):
                    frame_id = frame_ids[idx] if idx < len(frame_ids) else UUID_NIL
                    frame_queue.put_nowait(loop.create_task(extract_frame(cpu_frame_future, idx, frame_id)))
            except Exception as e:
                logging.error(f"Error in frame_callback: {e}")
                traceback.print_exc()
        def actual_frame_callback(*args):
            loop.call_soon_threadsafe(frame_callback, *args)

        gc.collect()
        torch.cuda.empty_cache()
        new_session = lambda: GenerationSession(
            params,
            config,
            frame_callback=actual_frame_callback,
            models=models
        )
        session = new_session()

        new_data_event = asyncio.Event()
        async def generate_loop():
            try:
                while True:
                    try:
                        await loop.run_in_executor(generate_pool, session.generate_block, models)
                    except asyncio.CancelledError:
                        # Generation completed all blocks
                        logging.info(f"Generation completed: {session.block_idx}/{session.num_blocks} blocks")
                        try:
                            # Only send if websocket is still connected
                            await websocket.send_json({"session_id": id, "status": "completed"})
                        except Exception as e:
                            logging.debug(f"Could not send completion message (websocket likely closed): {e}")
                        break
                    except Exception as e:
                        logging.error(f"Error during generation: {e}")
                        traceback.print_exc()
            except Exception as e:
                logging.error(f"Error in generate_loop: {e}")
        generate_task = loop.create_task(generate_loop())

        async for data in websocket.iter_bytes():
            frame = unpackb(data)
            if not isinstance(frame, dict):
                logging.warning(f"Received non-dict frame data: {frame}")
                continue
            if frame.get("action") == "reset":
                session.dispose()
                session = new_session()
            if frame.get("prompt", session.params.prompt) != session.params.prompt:
                params.prompt = frame["prompt"]
                try:
                    interp_steps = int(frame.get("interp_steps", frame.get("interpolation_steps", 4)))
                except Exception:
                    interp_steps = 4
                interp_steps = max(1, int(interp_steps))
                session.interpolate_prompt_embeds(models, session.params.prompt, interp_steps)
            if (new_seed := frame.get("seed", None)) is not None:
                session.params.seed = int(new_seed)
            if (image := frame.get("image")):
                await loop.run_in_executor(encode_pool, session.push_frame, image, frame.get("strength", None), frame.get("request_id", None))
                if (timestamp := frame.get("timestamp")) and isinstance(timestamp, (int, float)):
                    timestamp /= 1000.0
                    if time.time() - timestamp > 1.0:
                        logging.warning(f"High latency detected: {time.time() - timestamp:.2f}s")
            new_data_event.set()
    except WebSocketDisconnect:
        logging.info(f"Client disconnected from api session {id}, generation session {session.session_id if session else None}")
    finally:
        logging.info(f"Terminating session")
        if session:
            session.dispose()
        if frame_sender_task:
            frame_sender_task.cancel()
        if generate_task:
            generate_task.cancel()
        # Send session ID for video download
        try:
            await websocket.send_json({"session_id": id, "status": "completed"})
        except:
            pass

@app.websocket("/session/{id}")
async def app_session(websocket: WebSocket, id: str):
    return await ws_session(websocket, id, config=app.state.config, models=app.state.models)
