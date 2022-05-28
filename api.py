import torch

# from diffusion import *
from types import SimpleNamespace
from diffusion import (
    do_run,
    args,
    create_model_and_diffusion,
    split_prompts,
    model_config,
    model_path,
    diffusion_model,
)
from config import device
import gc

set_seed = 2586166778
if set_seed == "random_seed":
    random.seed()
    seed = random.randint(0, 2 ** 32)
    # print(f'Using seed: {seed}')
else:
    seed = int(set_seed)

text_prompts = {
    0: [
        "A beautiful painting of a singular lighthouse",
        "by greg rutkowski",
        "trending on artstation:-0.99",
    ]
}
raw_prompt = "a shining lighthouse on the shore of a tropical island in a raging storm | text:-0.99 | watermark:-0.99 | logo:-0.99 | watercolor"
if raw_prompt:
    # input_prompts = iargs.prompts.split('|')
    # for x in range(len(input_prompts)):
    #    input_prompts[x] = input_prompts[x].strip()

    text_prompts = {
        # 0: input_prompts,
        # 0: [iargs.prompts],
        0: raw_prompt.split("|"),
        # 100: ["This set of prompts start at frame 100", "This prompt has weight five:5"],
    }

    # not necessary but makes the cli output easier to parse
    for x in range(len(text_prompts[0])):
        text_prompts[0][x] = text_prompts[0][x].strip()


req = {
    "seed": seed,
    "prompts_series": split_prompts(text_prompts) if text_prompts else None,
    "output_filename": "demo.png",
    "output_dir": "output/",
}
args.update(req)

args = SimpleNamespace(**args)

print("Prepping model...")
model, diffusion = create_model_and_diffusion(**model_config)
model.load_state_dict(
    torch.load(f"{model_path}/{diffusion_model}.pt", map_location="cpu")
)
model.requires_grad_(False).eval().to(device)
for name, param in model.named_parameters():
    if "qkv" in name or "norm" in name or "proj" in name:
        param.requires_grad_()
if model_config["use_fp16"]:
    model.convert_to_fp16()

gc.collect()
torch.cuda.empty_cache()
try:
    # import pdb; pdb.set_trace()
    do_run(args, model, diffusion)
except KeyboardInterrupt:
    pass
finally:
    # print('Seed used:', seed)
    gc.collect()
    torch.cuda.empty_cache()

