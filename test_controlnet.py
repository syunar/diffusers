from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
import torch
from typing import Any, Union, Optional
from controlnet_aux import LineartAnimeDetector, LineartDetector
from transformers import CLIPTextModel
import cv2
from PIL import Image, ImageOps
import numpy as np
from torchvision.transforms import (
    Compose,
    ToTensor,
    Resize,
    functional
)
from compel import Compel

import torch
from torchvision.transforms import ToPILImage
import os
import random

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# BASE_MODEL = "runwayml/stable-diffusion-v1-5"
BASE_MODEL = "Linaqruf/anything-v3.0"
# BASE_MODEL = "xyn-ai/anything-v4.0"
# BASE_MODEL = "shibal1/anything-v4.5-clone"
# BASE_MODEL = "stablediffusionapi/anything-v5"
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15s2_lineart_anime"
# CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_lineart"
# CONTROLNET_MODEL = "ioclab/control_v1p_sd15_brightness"
# CONTROLNET_MODEL = "model_out_v1/checkpoint-5000/controlnet"

prompt = "1boy, 1girl, breasts, collared_shirt, comic, english_text, long_hair, multiple_girls, open_mouth, school_uniform, shirt, sitting, skirt"
# prompt = "1boy, 1girl, bar_censor, black_hair, blush, breasts, cardigan, censored, clothed_sex, clothes_lift, comic, cowgirl_position, cum, cum_in_pussy, doggystyle, ejaculation, english_text, hetero, long_hair, nipples, nude, open_mouth, oral, penis, pleated_skirt, school_uniform, sex, sex_from_behind, skirt, sweat, takarada_rikka, vaginal"
# prompt = "letterboxed, hetero, sex, comic, doggystyle, skirt, 1boy, cum, sex_from_behind, english_text, penis, 1girl, breasts, pleated_skirt, nipples, vaginal, takarada_rikka, school_uniform, long_hair, bar_censor, multiple_boys" "bar_censor, censored, oral, penis, fellatio, letterboxed, pointless_censoring, licking_penis, hetero, 1boy, school_uniform, identity_censor, blush, nose_blush, multiple_girls, comic, 2girls, short_hair, squatting, bow, partially_colored"
n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, horror_(theme), blood, speech_bubble, text"
guidance_scale = 7.5
controlnet_conditioning_scale = 1.2
eta = 1.0
PADDING = False
USE_COMPEL = True
CONTROLIMAGE_RESOLUTION = [512, 768, 890, 1024, None]
DEVICE = "cuda"

if BASE_MODEL == "runwayml/stable-diffusion-v1-5":
    CLIP_SKIP = 1
else:
    CLIP_SKIP = 2

controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=torch.float16)
processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
# controlnet = ControlNetModel.from_pretrained(, torch_dtype=torch.float16)
# vae = AutoencoderKL.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, subfolder="vae")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
text_encoder = CLIPTextModel.from_pretrained(
            BASE_MODEL,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
            num_hidden_layers=12 - (CLIP_SKIP - 1),
        )

# unet = UNet2DConditionModel.from_pretrained("/home/users/fronk/dev/diffusers/Anything_ink/Anything-Ink-NoVAE.safetensors", torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    # unet=unet,
    text_encoder=text_encoder,
    vae=vae,
    controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
).to(DEVICE)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

compel = Compel(
            tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, truncate_long_prompts=False
        )




#####

for control_res in CONTROLIMAGE_RESOLUTION:

    # generate image
    generator = torch.Generator(device=DEVICE).manual_seed(random.randint(0, 9999))


    # image = Image.open("018.jpg")
    image = Image.open("/home/users/fronk/projects/deepcolor/datasets/nhentai/228626/017.jpg")
    control_image = image.convert("L")

    w,h = control_image.size
    if control_res is None:
        control_res = min(control_image.size)

    control_image = processor(control_image, 
                                detect_resolution=min(control_image.size), 
                                image_resolution=control_res)

    class SquarePad:
        def __call__(self, image):
            max_wh = max(image.size)
            p_left, p_top = [(max_wh - s) // 2 for s in image.size]
            p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
            padding = (p_left, p_top, p_right, p_bottom)
            return functional.pad(image, padding, 0, 'constant')

    if PADDING:
        transform = Compose(
            [
                SquarePad(),
                Resize(control_res),
                ToTensor()
            ]
        )
    else:
        transform = Compose(
            [
                Resize(control_res),
                ToTensor()
            ]
        )


    # Create an instance of the ToPILImage transform
    to_pil = ToPILImage()

    
    control_image_pt = transform(control_image)
    control_image = to_pil(control_image_pt)
    # control_image = ImageOps.invert(to_pil(control_image_pt))
    control_image.save("lineart.png")
    print(control_image.size)
    if USE_COMPEL:
        prompt_embeds = compel.build_conditioning_tensor(prompt)
        negative_prompt_embeds = compel.build_conditioning_tensor(n_prompt)
        [prompt_embeds, negative_prompt_embeds] = compel.pad_conditioning_tensors_to_same_length(
            [prompt_embeds, negative_prompt_embeds]
        )
        image = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=20, 
            generator=generator, 
            image=control_image, 
            num_images_per_prompt=2,
            eta=eta,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ).images
    else:

        image = pipe(
            prompt=prompt,
            n_prompt=n_prompt,
            num_inference_steps=20, 
            generator=generator, 
            image=control_image, 
            num_images_per_prompt=2,
            eta=eta,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ).images



        


    # image.append(control_image)

    image_grid(image, 1, len(image)).save(f"log_{control_res}.png")

