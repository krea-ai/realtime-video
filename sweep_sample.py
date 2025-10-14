# %% [markdown]
import os
import glob

from safetensors.torch import load_file, save_file
model_paths = glob.glob("release_candidates_sf_14b/*")
def load_ckpt(path):
    sft_path = path.replace(".pt", ".sft")
    if os.path.exists(sft_path):
        path = sft_path
    if path.endswith(".safetensors") or path.endswith(".sft") :
        print("Loading safetensors file", path)
        state_dict = load_file(path, device="cuda")
    else:
        print("Loading pt file", path)
        state_dict = torch.load(path, map_location="cuda")
        save_file(state_dict, sft_path)
    state_dict = {
        "model." + k if not k.startswith("model.") else k: v for k, v in state_dict.items()
    }
    return state_dict
# print("model paths:", model_paths)
model_paths = [
    "checkpoints/14b_200.safetensors",
    # "/workspace/self-forcing/checkpoints/lightx2v_stepdistill_bidirectional14b.sft",
    # "/workspace/self-forcing/checkpoints/lightx_14b_causal_3000.sft",
    # "/workspace/sf-copy/merged_checkpoints_v2/merged_0.1_0.9_0.0.safetensors",
    # "/workspace/sf-copy/merged_checkpoints_v4/merged_0.05_0.0_0.95.safetensors", 
    # "/workspace/sf-copy/merged_checkpoints_v4/merged_0.05_0.95_0.0.safetensors", 
    # "/workspace/sf-copy/merged_checkpoints_v4/merged_0.05_0.8_0.15.safetensors",
    # "/workspace/sf-copy/release_candidates_sf_14b/merged_0.1_0.0_0.9_bf16.sft",
]

# # Video Sampling Example
# 
# This notebook demonstrates how to use the sampling functions to generate videos from text prompts.

# %%
# import sys
# sys.path.insert(0, '/workspace/sf-copy')

from sample import sample_videos, sample_single_video, prompts
from release_server import load_merge_config, load_all

# %% [markdown]
# ## Option 1: Load models once and reuse (Recommended)
# 
# This approach loads the models once and reuses them for multiple prompts, which is much more efficient.

# %%
# Load models once
config = load_merge_config("configs/self_forcing_server_14b.yaml")
config.checkpoint_path = model_paths[0]

models = load_all(config)

print("✅ Models loaded and ready to use!")

# %% [markdown]
# ## Generate videos for multiple prompts

# %%
from sample import GenerateParams
params = GenerateParams(
    prompt="A cat playing with a ball of yarn",
    width=832,
    height=480,
    num_blocks=8,
    is_t2v=True,
    seed=123,
    kv_cache_num_frames = 3,
    keep_first_frame = False,
    thresh_kv_scale = 1.,
    num_denoising_steps = 8,

)

# %%
import importlib
import sample
importlib.reload(sample)


# %%
test_prompts = [
    "A sorcerer stands with one hand outstretched, holding a roiling flame that coils and twists restlessly around his palm and fingers, sparks shooting off in unpredictable arcs. The fire spirals and lashes outward, wrapping around his arm like a living serpent before snapping back toward his hand in a continuous, fluid motion. His cloak ripples in the heat’s updraft as the flame flares and contracts, creating bursts of glowing embers that rise and scatter through the air",
    "Adrenaline-pumped, wide-eyed ginger kitten in vintage aviator goggles blasts down a narrow cobblestone street on a bright-yellow mini-bicycle. Camera whip-pans in from behind, then snap-zooms past the spinning front wheel into an ultra-close-up of the cat’s determined face, fur rippling in the wind. Hard sunlight streaks across the scene, creating dynamic highlights and streaking motion-blur on the spokes.",
    "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.",
    "A lone samurai in traditional armor practices kata in a quiet field at dawn, his silhouette framed against a misty horizon. Each sword strikeoad is fluid yet forceful, the blade flashing as it slices through the air with precise arcs and sudden bursts of speed. His feet shift with practiced agility, sending up small sprays of dirt as he pivots, steps, and lunges in a continuous flow of disciplined motion. The static camera holds the scene firmly in place, emphasizing the intensity and grace of his movements against the stillness of the landscape.",
    "Tracking shot, cinematic style, of a surreal alien bird with iridescent feathers and bio-luminescent wings, gracefully flying through a dense alien forest — the camera follows the bird in one continuous dynamic shot, weaving through twisted glowing trees, over bioluminescent mushrooms, and past floating pollen-like orbs. The bird performs elegant aerial maneuvers — barrel rolls, sharp turns, dives — as it dodges branches and interacts with the strange atmosphere. The forest pulses with ambient light and motion-reactive flora.",
    "A single crystalline drop of water hovers in midair against a soft gradient background, shimmering with refracted light. The drop quivers, rippling with internal motion, before stretching outward as if blooming from within. Its surface tension warps into delicate translucent petals that unfurl in one continuous motion. The petals radiate with glistening reflections, the drop gracefully becoming a luminous flower suspended in space.",
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage",
    "She returns in a radiant gown made entirely of flowing gold silk, draped in voluminous layers that trail behind her in shimmering waves. The fabric ripples dramatically with each step, catching and scattering light across the runway. The dress moves like liquid metal, transforming her walk into a continuous display of luxury and opulence, radically different from her previous sculptural look.",
    "Inside a steaming porcelain coffee cup, two tiny wooden galleons clash on dark, swirling waves of coffee. Cannons fire with sharp flashes, sending miniature plumes of smoke curling into the air, while splinters of wood spray from direct hits. The surface of the coffee sloshes violently with every broadside, waves lapping against the cup’s rim. Steam drifts upward, blending with the smoke, as the ships circle each other in a chaotic, continuous duel.",
    "A massive wooden ship sails through a dark, stormy sea, its masts swaying violently as enormous waves crash against the hull. The ocean roars with relentless power, sending bursts of white spray over the deck as the crew struggles to keep control. Scene from an award-winning movie, Underexposed, undersaturated, muted color palette, cinematic scene with professional color grading", 
    "In front of a western saloon, a cowboy spins a revolver then turns quickly and shoots it, a wisp of smoke coming from the barrel",
    "From within the blaze, the sorcerer’s form mutates as enormous wings of fire burst from his back, each feather a living tongue of flame. The wings unfurl with explosive force, showering the night with embers. He beats them once, sending a wave of heat rippling through the air that bends the storm clouds above. The ground shakes as glowing rocks tumble away from the platform, cast into the darkness by the surging winds.",
    "A cyborg bird flaps its wings and takes flight, its metallic feathers glinting in the sunlight. The bird soars through the sky, its body transforming into a sleek, aerodynamic shape as it glides effortlessly through the air. The bird's eyes glow with an inner light, and its wings spread wide to catch the wind. The bird's body is made of a combination of organic and synthetic materials, with a sleek, modern design that allows it to move with ease through the air. The bird's body is made of a combination of organic and synthetic materials, with a sleek, modern design that allows it to move with ease through the air.",
    "A single purple flower rests against a deep black background, its closed bud trembling slightly before beginning to unfurl. In a smooth timelapse motion, the petals peel outward layer by layer, stretching and curving as they catch subtle highlights that shimmer along their velvety edges. The bloom expands fully, revealing the intricate textures and radiant color of the flower in one continuous, fluid motion.",
    "Abstract stunning visuals of smoke morphing into a face, which then turns back into smoke, looping video, cinematic style, muted color palette, soft lighting, dynamic motion.",
    

]

# %%
print(test_prompts[:2][0])

# %%
import importlib
import sample
importlib.reload(sample)

import sample
from safetensors.torch import load_file
import glob
import torch

# Sample videos for all prompts with custom settings

# model_paths = glob.glob("merged_checkpoints/*")


for i, model_path in enumerate(model_paths):
    output_dir = f"merge_sweep_outputs_v6_no_reencode/{os.path.basename(model_path)}"
    if i != 0:
        print("loading model", model_path)
        if os.path.exists(output_dir):
            print("skipping", output_dir)
            continue
        sd = load_ckpt(model_path)
        m, u = models.pipeline.generator.load_state_dict(sd, strict=False)
    for block in models.pipeline.generator.model.blocks:
        block.self_attn.fused_projections = False
    
    # print("model loaded, missing:", m, "unexpected:", u)
    if i != 0:
        del sd
    import gc
    gc.collect()

    results = sample.sample_videos(
        prompts_list=test_prompts,
        models=models,  # Reuse loaded models
        params=params,
        output_dir=output_dir,
        save_videos=True,
        fps=16
    )
    torch.cuda.empty_cache()

# Display results