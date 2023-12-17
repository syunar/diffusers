from PIL import Image
import os
import re

import torch
import numpy as np
import PIL.Image
from typing import Optional, Union, List
import subprocess

from pathlib import Path

from model import DeepDanbooruModel

LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def resize_image(resize_mode, im, width, height):
    """
    Resizes an image with the specified resize_mode, width, and height.

    Args:
        resize_mode: The mode to use when resizing the image.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        im: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
        upscaler_name: The name of the upscaler to use. If not provided, defaults to opts.upscaler_for_img2img.
    """

    def resize(im, w, h):
        return im.resize((w, h), resample=LANCZOS)

    if resize_mode == 0:
        res = resize(im, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if fill_height > 0:
                res.paste(
                    resized.resize((width, fill_height), box=(0, 0, width, 0)),
                    box=(0, 0),
                )
                res.paste(
                    resized.resize(
                        (width, fill_height),
                        box=(0, resized.height, width, resized.height),
                    ),
                    box=(0, fill_height + src_h),
                )
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if fill_width > 0:
                res.paste(
                    resized.resize((fill_width, height), box=(0, 0, 0, height)),
                    box=(0, 0),
                )
                res.paste(
                    resized.resize(
                        (fill_width, height),
                        box=(resized.width, 0, resized.width, height),
                    ),
                    box=(fill_width + src_w, 0),
                )

    return res


class DeepDanbooru:
    """
    # example#1: how to inference
    ddbr = DeepDanbooru()
    ddbr.tag(pil_image=Image.open("/content/result.jpg"), deepbooru_filter_tags="monochrome")

    # example#2: list all the tags we have
    ddbr.model.tags
    pattern = r'multi'
    [string for string in ddbr.model.tags if re.match(pattern, string)]
    """

    def __init__(self, device: Optional[str] = "cpu"):
        self.model = None
        self.files = Path(f"{os.path.expanduser('~')}/.cache/dmc/models/model-resnet_custom_v3.pt")
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load()

    def load(self):
        if self.model is not None:
            return

        if os.path.exists(self.files):
            print(f"The file '{self.files}' exists.")
        else:
            print(f"The file '{self.files}' does not exist.")
            print(f"Downloading..")
            os.makedirs(os.path.dirname(self.files), exist_ok=True)
            subprocess.run(
                [
                    "wget",
                    "-v",
                    "https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt",
                    "-O",
                    self.files,
                ]
            )

        self.model = DeepDanbooruModel()
        self.model.load_state_dict(torch.load(self.files, map_location="cpu"))

        self.model.eval()
        self.model.to(self.device)

    def post_process(self, deepbooru_filter_tags: str, y: np.ndarray):

        threshold = 0.5
        use_spaces = True
        use_escape = True
        alpha_sort = True
        include_ranks = False and not force_disable_ranks
        probability_dict = {}

        for tag, probability in zip(self.model.tags, y):
            if probability < threshold:
                continue

            if tag.startswith("rating:"):
                continue

            probability_dict[tag] = probability

        if alpha_sort:
            tags = sorted(probability_dict)
        else:
            tags = [tag for tag, _ in sorted(probability_dict.items(), key=lambda x: -x[1])]

        res = []

        filtertags = {x.strip().replace(" ", "_") for x in deepbooru_filter_tags.split(",")}

        for tag in [x for x in tags if x not in filtertags]:
            re_special = re.compile(r"([\\()])")
            probability = probability_dict[tag]
            tag_outformat = tag
            if use_spaces:
                tag_outformat = tag_outformat.replace("_", " ")
            if use_escape:
                tag_outformat = re.sub(re_special, r"\\\1", tag_outformat)
            if include_ranks:
                tag_outformat = f"({tag_outformat}:{probability:.3f})"

            res.append(tag_outformat)

        return ", ".join(res)

    def __call__(
        self,
        pil_image: Union[PIL.Image.Image, List[str], torch.Tensor],
        deepbooru_filter_tags: Optional[str] = "",
        force_disable_ranks: bool = False,
    ) -> str:
        # /workspaces/stable-diffusion-webui/modules/shared_options.py:210


        # pic = resize_image(2, pil_image.convert("RGB"), 512, 512)
        # pic.save("pic.png")

        # a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255
        x = pil_image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            # x = torch.from_numpy(a).to(self.device)
            y = self.model(x)[0].detach().cpu().numpy()

        


        return self.post_process(deepbooru_filter_tags, y)

        
