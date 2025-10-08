# %% [markdown]
import os
# # Video Sampling Example
# 
# This notebook demonstrates how to use the sampling functions to generate videos from text prompts.

# %%
import sys
sys.path.insert(0, '/workspace/sf-copy')

from sample import sample_videos, sample_single_video, prompts
from release_server import load_merge_config, load_all

# %% [markdown]
# ## Option 1: Load models once and reuse (Recommended)
# 
# This approach loads the models once and reuses them for multiple prompts, which is much more efficient.

# %%
# Load models once
config = load_merge_config("configs/self_forcing_server_14b.yaml")
# config.checkpoint_path = "/workspace/sf-copy/checkpoints/self_forcing_dmd.pt"

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
    num_blocks=10,
    is_t2v=True,
    seed=12345,
    kv_cache_num_frames=3,
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
    "A lone samurai in traditional armor practices kata in a quiet field at dawn, his silhouette framed against a misty horizon. Each sword strike is fluid yet forceful, the blade flashing as it slices through the air with precise arcs and sudden bursts of speed. His feet shift with practiced agility, sending up small sprays of dirt as he pivots, steps, and lunges in a continuous flow of disciplined motion. The static camera holds the scene firmly in place, emphasizing the intensity and grace of his movements against the stillness of the landscape.",
    "Tracking shot, cinematic style, of a surreal alien bird with iridescent feathers and bio-luminescent wings, gracefully flying through a dense alien forest — the camera follows the bird in one continuous dynamic shot, weaving through twisted glowing trees, over bioluminescent mushrooms, and past floating pollen-like orbs. The bird performs elegant aerial maneuvers — barrel rolls, sharp turns, dives — as it dodges branches and interacts with the strange atmosphere. The forest pulses with ambient light and motion-reactive flora.",
    "A single crystalline drop of water hovers in midair against a soft gradient background, shimmering with refracted light. The drop quivers, rippling with internal motion, before stretching outward as if blooming from within. Its surface tension warps into delicate translucent petals that unfurl in one continuous motion. The petals radiate with glistening reflections, the drop gracefully becoming a luminous flower suspended in space.",
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage",
    "She returns in a radiant gown made entirely of flowing gold silk, draped in voluminous layers that trail behind her in shimmering waves. The fabric ripples dramatically with each step, catching and scattering light across the runway. The dress moves like liquid metal, transforming her walk into a continuous display of luxury and opulence, radically different from her previous sculptural look.",
    "Inside a steaming porcelain coffee cup, two tiny wooden galleons clash on dark, swirling waves of coffee. Cannons fire with sharp flashes, sending miniature plumes of smoke curling into the air, while splinters of wood spray from direct hits. The surface of the coffee sloshes violently with every broadside, waves lapping against the cup’s rim. Steam drifts upward, blending with the smoke, as the ships circle each other in a chaotic, continuous duel.",
    "A massive wooden ship sails through a dark, stormy sea, its masts swaying violently as enormous waves crash against the hull. The ocean roars with relentless power, sending bursts of white spray over the deck as the crew struggles to keep control."
    # "In a brightly lit gymnasium, a young gymnast in a red leotard swings powerfully on the uneven bar, launching into a perfect backflip. Her body tucks tightly mid-air, spinning in one continuous motion before her hands snap back onto the bar with precision. Chalk dust bursts into the air with the impact, drifting like smoke around her. The crowd gasps faintly in the background, the scene locked in graceful, dynamic motion.",
    # "Under glowing neon lights in a surreal diner, a polar bear sits awkwardly in a red leather booth, gripping a massive BLT with mayonnaise dripping from its paws. It devours the sandwich with primal hunger, grease smearing across its white fur as the jukebox hums in the background. Fries scatter across the floor as the bear’s huge body shifts against the tiny furniture, the booth creaking under its weight. The continuous shot blends humor, absurdity, and menace in equal measure.",
    # "Under the glare of bright stadium lights, the golden retriever launches high off the springboard, twisting into a daring double flip. Its ears flap wildly mid-air, and the shimmering pool reflects its spinning form. The audience erupts in cheers as the dog’s paws extend gracefully in the final moment before the dive. A splash explodes upward as it enters the water, sending ripples across the Olympic rings painted on the pool floor.", 
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
model_paths = glob.glob("release_candidates_14b/*")
# print("model paths:", model_paths)
# model_paths = [
#     "/workspace/sf-copy/merged_checkpoints_v2/merged_0.1_0.9_0.0.safetensors",
#     "/workspace/sf-copy/merged_checkpoints_v2/merged_0.1_0.0_0.9.safetensors",
# ]



for i, model_path in enumerate(model_paths):
    print("loading model", model_path)
    output_dir = f"merge_sweep_outputs_v7_8step/{os.path.basename(model_path)}"
    if os.path.exists(output_dir):
        print("skipping", output_dir)
        continue
    sd = load_file(model_path)
    sd = {
        "model." + k if not k.startswith("model.") else k: v for k, v in sd.items()
    }
    m, u = models.pipeline.generator.load_state_dict(sd, strict=False)
    for block in models.pipeline.generator.model.blocks:
        block.self_attn.fused_projections = False
    
    # print("model loaded, missing:", m, "unexpected:", u)
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