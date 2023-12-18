import pandas as pd
import glob

## get image dirs
img_dir_list = glob.glob(r"validation_images/228626/*.jpg")

df = pd.DataFrame(img_dir_list, columns=["filename"])

## init model
from deepdanbooru import DeepDanbooruModel
from squarepad import SquarePad
import torch
from torchvision import transforms
import numpy as np
from typing import List
from PIL import Image
from tqdm.auto import tqdm

model = DeepDanbooruModel()
model.load_state_dict(torch.load("model-resnet_custom_v3.pt", map_location="cpu"))
model = torch.compile(model)
id2label = {k:v for k,v in enumerate(model.tags)}
label2id = {v:k for k,v in enumerate(model.tags)}
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

def mapping_output_tensor_to_tags(output: np.ndarray, 
                                  id2label: dict, 
                                  threshold: float = 0.5, 
                                  sorted: bool=False) -> List[str]:
    # output = outputs.detach().cpu().numpy()
    prefixes_to_ignore = ["rating:", "letterboxed", "black_border", "monochrome", "greyscale"]
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

transform = transforms.Compose(
    [
        SquarePad(),
        transforms.Resize(512),
        transforms.ToTensor()
    ]
)

captions = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    filename = row["filename"]
    x = Image.open(filename).convert("RGB")
    x = transform(x).unsqueeze(0).to(device)
    y = model(x).detach().cpu().numpy()

    caption = mapping_output_tensor_to_tags(output=y,
                                            id2label=id2label,
                                            threshold=0.5,
                                            sorted=False)
    captions.extend(caption)

df["captions"] = captions

del model
import gc
torch.cuda.empty_cache()
gc.collect()

df.to_csv("validation_images/captions.csv", index=False)

'" "'.join(df.iloc[:5]["filename"].to_list())
'" "'.join(df.iloc[:5]["captions"].to_list())