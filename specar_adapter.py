"""SpecAR for Krea Realtime (https://github.com/statite-ai/SpecAR): the 14B model generates `specar_k` evenly spaced
blocks of a session and the Self-Forcing 1.3B drafter generates the rest.

Off by default. It turns on when SPECAR_DRAFTER_CKPT points to the drafter checkpoint (self_forcing_dmd.pt) and the
`specar` package is installed. A session opts in by sending `specar_k`; without it, or when SpecAR is not available,
the session runs the stock path, which this module does not touch.
"""
import asyncio
import logging
import os
import threading
import types
import weakref

import torch

log = logging.getLogger(__name__)
STATE = {"enabled": False, "reason": "SPECAR_DRAFTER_CKPT is not set", "pending_bytes": 0}
# Free GPU memory a new SpecAR session needs beyond its own caches: the working memory of a 14B block (about 8 GiB
# measured) and the encoding of a start frame or input video.
HEADROOM_BYTES = 9 * 2**30
# Sessions that were disposed while still holding their caches. The generation worker frees them before it runs the
# next SpecAR block, so their memory counts as free for a new session (reset and reconnect create one immediately).
_DISPOSED = weakref.WeakSet()


def _reap(current=None):
    """Free the caches of disposed sessions. Runs on the generation worker, so no block of theirs is executing."""
    for session in list(_DISPOSED):
        if session is not current:
            session._release()
            _DISPOSED.discard(session)


def _off(reason):
    STATE.update(enabled=False, reason=reason)
    log.error("SpecAR disabled: %s", reason)


def setup(models):
    """Load the drafter. Call once after the models are loaded and BEFORE they are compiled: SpecAR sessions run the
    uncompiled modules, so torch.compile of the stock path is left as it is."""
    ckpt = os.getenv("SPECAR_DRAFTER_CKPT")
    if not ckpt:
        return
    if not os.path.exists(ckpt):
        return _off(f"drafter checkpoint not found: {ckpt}")
    try:
        from specar import model_krea
    except Exception as e:
        return _off(f"the specar package cannot be imported ({e}); pip install specar")
    try:
        backend = model_krea._install_attention()      # FlashAttention 3 on Hopper, 4 on Blackwell
    except Exception as e:
        backend = None
        log.warning("SpecAR: keeping Krea's attention path (%s)", e)
    try:
        drafter = model_krea.load_wan_wrapper(ckpt)
    except Exception as e:
        return _off(f"could not load the drafter from {ckpt}: {e}")
    STATE.update(enabled=True, reason="", backend=backend, engine=model_krea,
                 models=types.SimpleNamespace(target=models.transformer, drafter=drafter, text_encoder=models.text_encoder,
                                              vae_encoder=models.vae_encoder, vae_decoder=models.vae_decoder))
    log.info("SpecAR enabled (drafter %s, attention %s)", ckpt, backend)


def session_bytes(params):
    """GPU memory one SpecAR session holds: the KV and cross-attention caches of both models (bf16)."""
    tokens, total = (params.kv_cache_num_frames + 3) * 1560, 0
    for wrapper in (STATE["models"].target, STATE["models"].drafter):
        total += len(wrapper.model.blocks) * 2 * (tokens + 512) * wrapper.model.config.dim * 2
    return total


def free_bytes():
    """Free GPU memory, counting what PyTorch holds but is not using, minus sessions that have not allocated yet."""
    releasing = sum(s._cache_bytes for s in list(_DISPOSED) if s._session is not None)
    return (torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            - STATE["pending_bytes"] + releasing)


def make_session(stock_cls, session_counter, params, config, frame_callback, models):
    """A SpecAR session when the request asks for one and SpecAR can serve it; otherwise the stock session."""
    k = getattr(params, "specar_k", None)
    if k:
        why = None if STATE["enabled"] else STATE["reason"]
        if why is None and getattr(params, "webcam_mode", False):
            why = "webcam mode is not supported"
        if why is None:
            need, free = session_bytes(params), free_bytes()
            if free < need + HEADROOM_BYTES:
                why = (f"not enough free GPU memory for another SpecAR session: {free / 2**30:.1f} GiB free, "
                       f"{(need + HEADROOM_BYTES) / 2**30:.1f} GiB needed")
        if why is None:
            return SpecARGenerationSession(params, frame_callback, session_counter, need)
        log.warning("specar_k=%s ignored (%s); running the stock path", k, why)
    return stock_cls(params, config, frame_callback=frame_callback, models=models)


class SpecARGenerationSession:
    """The part of GenerationSession's interface the server uses. Constructing it is free: all GPU work, including
    building the SpecAR session with its own KV and cross-attention caches, happens in generate_block, i.e. on the
    single generation worker, so sessions never touch shared model state concurrently."""

    def __init__(self, params, frame_callback, session_counter, reserve_bytes=0):
        self._reserved = reserve_bytes              # counted against free memory until the caches exist
        self._cache_bytes = reserve_bytes
        STATE["pending_bytes"] += reserve_bytes
        self._finished = False
        self._latents = None
        self.params = params
        self.frame_callback = frame_callback
        self.session_id = session_counter.increment()
        self.gpu = torch.cuda.current_device()
        self.block_idx = 0
        self.num_blocks = params.num_blocks
        self.disposed = threading.Event()
        self._session = None
        self._pending_prompt = None       # set by the server thread, applied by the generation worker
        self._interpolated = []

    def dispose(self):
        self.disposed.set()
        _DISPOSED.add(self)

    def interpolate_prompt_embeds(self, models, new_prompt, interpolation_steps):
        self._pending_prompt = (new_prompt, interpolation_steps)

    def push_frame(self, *args, **kwargs):
        log.warning("webcam frames are ignored by SpecAR sessions")

    def _unreserve(self):
        STATE["pending_bytes"] -= self._reserved
        self._reserved = 0

    def _release(self):
        """Free the session's caches (on the generation worker) as soon as it has ended."""
        if self._session is not None:
            self._latents = self._session.all_latents[:, :self._session.current_start_frame].clone()
            self._session = None
        self._unreserve()
        self._finished = True

    def __del__(self):
        self._unreserve()

    def _start(self):
        from specar.schedule import uniform_k
        p, k = self.params, int(self.params.specar_k)
        emit = self.frame_callback          # not `self`: the engine session must not keep this object alive
        self._session = STATE["engine"].SpecARSession(
            STATE["models"], p.prompt, p.seed, p.num_blocks, lambda h: uniform_k(min(k, h), h),
            width=p.width, height=p.height, kv_cache_num_frames=p.kv_cache_num_frames,
            num_denoising_steps=p.num_denoising_steps, timestep_shift=p.timestep_shift, strength=p.strength,
            keep_first_frame=p.keep_first_frame, start_image=p.start_frame, input_video=p.input_video,
            frame_callback=lambda pixels, event: emit(pixels, [], event))
        self._unreserve()
        self.num_blocks = self._session.num_blocks
        self.anchors = sorted(self._session.anchors)

    def _apply_prompt(self):
        s = self._session
        if self._pending_prompt is not None:          # the blend of GenerationSession.interpolate_prompt_embeds
            (prompt, steps), self._pending_prompt = self._pending_prompt, None
            e1 = s.current_prompt_embeds
            e2 = STATE["models"].text_encoder(text_prompts=[prompt])["prompt_embeds"].to(dtype=torch.bfloat16, device=e1.device)
            x = torch.lerp(e1, e2, torch.linspace(0, 1, steps=steps).unsqueeze(1).unsqueeze(2).to(e1))
            self._interpolated = list(x.chunk(steps, dim=0))
        if self._interpolated:
            e = self._interpolated.pop(0)
            s.set_prompt_embeds(e.to(dtype=s.current_prompt_embeds.dtype, device=s.current_prompt_embeds.device))

    @torch.inference_mode()
    def generate_block(self, models):
        with torch.cuda.device(self.gpu):
            _reap(current=self)
            if self.disposed.is_set() or self._finished:
                self._release()
                raise asyncio.CancelledError()
            if self._session is None:
                try:
                    self._start()
                except Exception as e:
                    self._session = None
                    self._release()
                    torch.cuda.empty_cache()
                    log.error("SpecAR session %s could not start (%s: %s); session ended", self.session_id, type(e).__name__, e)
                    raise asyncio.CancelledError()
            try:
                self._apply_prompt()
                pixels = self._session.generate_block()
            except Exception:
                # The server retries a failing block forever; end this session instead, with its caches freed.
                log.exception("SpecAR session %s failed at block %s; session ended", self.session_id, self.block_idx)
                self._release()
                raise asyncio.CancelledError()
            self.block_idx = self._session.block_idx
            if self.block_idx >= self.num_blocks:
                self._release()                 # the last block is out: free the caches now, the socket may stay open
            return pixels

    @property
    def all_latents(self):
        if self._session is None:
            return self._latents
        return self._session.all_latents[:, :self._session.current_start_frame]

    def __hash__(self):
        return id(self)
