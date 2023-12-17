# wget https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt

import glob
import os
from datasets import load_dataset, load_from_disk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import (
    Compose,
    ToTensor,
    Resize,
    ToPILImage
)

from torch.utils.data import DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import datasets
from datasets import Dataset, Features, Value
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from typing import List

from tqdm.auto import tqdm
from pathlib import Path
import multiprocessing as mp

from deepdanbooru import DeepDanbooruModel
from squarepad import SquarePad

import logging


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

MAX_IMAGES = 999999999
# SAVE_DIR = "data/test"
SAVE_DIR = "/home/users/fronk/projects/deepcolor/datasets/greyscale_manga_fullsize"
BATCH_SIZE = 256
MIN_RESOLUTION = 600

def generate_datasets():


    logging.basicConfig(
        level=logging.INFO,  # Set the desired logging level (e.g., INFO, WARNING, ERROR)
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),  # Log to a file
            logging.StreamHandler()          # Log to the console
        ]
    )

    base_directory = "/home/users/fronk/projects/deepcolor/colorize-download/data/"
    pattern = os.path.join(base_directory, "*", "*.[jJ][pP][gG]")

    all_images_dir_nhentai = list(glob.iglob(pattern))

    base_directory = "/home/users/fronk/projects/deepcolor/kaggle_datasets/color_full/"
    pattern = os.path.join(base_directory, "*.[jJ][pP][gG]")

    all_images_dir_kaggle = list(glob.iglob(pattern))

    logging.info(f"nhentai: {len(all_images_dir_nhentai)}, kaggle: {len(all_images_dir_kaggle)}")

    # all_images_dir = all_images_dir_nhentai
    all_images_dir = all_images_dir_kaggle + all_images_dir_nhentai

    logging.info(f"total: {len(all_images_dir)}")


    logging.info("Create Datasets...")


    def generate_entries(shards):
        index = 0
        for filename in shards:
            try:
                # print(image)
                img = Image.open(filename).convert("RGB")
                w, h = img.size

                if w == h:
                    if w < MIN_RESOLUTION:
                        continue
                if w > h:
                    aspect = w / h
                    if (aspect > 1.7) or (h < MIN_RESOLUTION):
                        continue
                if w < h:
                    aspect = h / w
                    if (aspect > 1.7) or (w < MIN_RESOLUTION):
                        continue

                yield {
                        "image": img,
                        "filename": filename,
                        "width": w,
                        "height": h
                    }
                
            except Exception as e:
                logging.info(f"Error processing image {filename}: {e}")
                continue  # Skip to the next file

            index += 1
            if index >= MAX_IMAGES:
                break
            
    ds = Dataset.from_generator(generate_entries, 
                                cache_dir="./.cache",
                                num_proc=mp.cpu_count(),
                                writer_batch_size=1000,
                                features=Features({"image": datasets.Image(),
                                                   "filename": Value(dtype='string', id=None),
                                                   "width": Value(dtype='int16', id=None),
                                                   "height": Value(dtype='int16', id=None)}),
                                gen_kwargs={"shards": all_images_dir[:MAX_IMAGES]}
                                # gen_kwargs={"shards": ['/home/users/fronk/projects/deepcolor/datasets/nhentai/228626/017.jpg']}
                                )


    return ds

def save_to_disk(ds, save_dir):
    logging.info("Save datasets to disk...")
    ds.save_to_disk(save_dir, max_shard_size="1GB")

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])

    return {
        "pixel_values": pixel_values
    }

def mapping_output_tensor_to_tags(output: np.ndarray, 
                                  id2label: dict, 
                                  threshold: float = 0.5, 
                                  sorted: bool=False) -> List[str]:
    # output = outputs.detach().cpu().numpy()
    prefixes_to_ignore = ["rating:", "letterboxed", "black_border"]
    if sorted:
        sorted_indices = np.argsort(output, axis=1)[:, ::-1]
        sorted_probs = np.take_along_axis(output, sorted_indices, axis=1)

        outputs_tags = []

        for probs_row, indices_row in zip(sorted_probs, sorted_indices):
            tags = [id2label[idx] for idx, prob in zip(indices_row, probs_row) if prob > threshold and not any(id2label[idx].startswith(prefix) for prefix in prefixes_to_ignore)]
            outputs_tags.append(", ".join(tags))

    else:
        outputs_tags = []

        for probs_row in output:
            tags = [id2label[idx] for idx, prob in enumerate(probs_row) if prob > threshold and not any(id2label[idx].startswith(prefix) for prefix in prefixes_to_ignore)]
            outputs_tags.append(", ".join(tags))

    return outputs_tags


def get_caption(ds):

    transform = Compose(
        [
            SquarePad(),
            Resize(512),
            ToTensor()
        ]
    )

    def image_tagging_transforms(examples):
        examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
        return examples
    
    processed_ds = ds.with_transform(image_tagging_transforms)
    dataloader = DataLoader(processed_ds, 
                            batch_size=BATCH_SIZE, 
                            collate_fn=collate_fn, 
                            shuffle=False)

    batch = next(iter(dataloader))

    for k,v in batch.items():
        logging.debug(k,v.shape)
    
    model = DeepDanbooruModel()
    model.load_state_dict(torch.load("model-resnet_custom_v3.pt", map_location="cpu"))
    model = torch.compile(model)
    id2label = {k:v for k,v in enumerate(model.tags)}
    label2id = {v:k for k,v in enumerate(model.tags)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    captions = []

    with torch.inference_mode():
        for idx, batch in enumerate(tqdm(dataloader)):
            batch = {k:v.to(device) for k,v in batch.items()}
            # ToPILImage()(batch["pixel_values"][0]).save("testA.png")
            outputs = model(batch["pixel_values"]).detach().cpu().numpy()
            captions.extend(mapping_output_tensor_to_tags(outputs, id2label=id2label, threshold=0.5))
            
    
    del model
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    ds = ds.add_column("text", captions)

    return ds

def main():
    # ds = generate_datasets()
    ds = load_from_disk(SAVE_DIR)
    ds = get_caption(ds)
    logging.info(ds[0])
    # save_to_disk(ds, SAVE_DIR)
    save_to_disk(ds, SAVE_DIR+"_with_caption")


if __name__ == "__main__":
    main()


# from datasets import load_dataset, load_from_disk
# ds = load_from_disk("/home/users/fronk/projects/deepcolor/datasets/greyscale_manga_fullsize_500k")

# import pandas as pd

# df = pd.DataFrame({
#     "filename": ds["filename"],
#     "width": ds["width"],
#     "height": ds["height"]
# })


# df[["width", "height"]].plot.scatter(x="width", y="height")

# df[df["width"] > 8000]["filename"].values[0]

# from PIL import Image

# Image.open('/home/users/fronk/projects/deepcolor/colorize-download/data/163234/036.jpg').size

# df.info()

# df.describe()

# df["width"].quantile(0.1)
# df["height"].quantile(0.1)
# df["width"].quantile(0.01), df["height"].quantile(0.01)


# df["aspect"] = df["height"] / df["width"]

# df

# df.describe()

# df["aspect"].quantile(0.05), df["aspect"].quantile(0.85)


# df["aspect"].hist()

# where = df["aspect"] > 4

# df[where]["filename"].tolist()[0]

# Image.open(df[where]["filename"].tolist()[0])

# '1boy, black_border, black_hair, breasts, collared_shirt, comic, english_text, head_rest, long_hair, monochrome, multiple_girls, school_uniform, shirt, sitting, skirt, speech_bubble'
# '1boy, 2girls, black_hair, breasts, collared_shirt, comic, english_text, long_hair, multiple_girls, open_mouth, school_uniform, shirt, sitting, skirt, speech_bubble'
    
# ['1boy, 1girl, blush, bow, bowtie, closed_eyes, clothed_sex, comic, doggystyle, english_text, fingering, hand_in_panties, hetero, monochrome, open_mouth, panties, plaid_skirt, pussy_juice, school_uniform, short_hair, skirt, striped, underwear, wet, wet_panties']
# ['1boy, 1girl, blush, bow, bowtie, closed_eyes, clothed_sex, comic, doggystyle, english_text, fingering, hand_in_panties, hetero, monochrome, open_mouth, panties, pussy_juice, school_uniform, short_hair, skirt, striped, underwear, wet, wet_clothes, wet_panties']