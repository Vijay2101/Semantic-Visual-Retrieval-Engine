import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------
# Load Model
# ---------------------------------------
print("Loading model...")
model = CLIPModel.from_pretrained("clip-rsicd").to(DEVICE)
processor = CLIPProcessor.from_pretrained("clip-rsicd")
model.eval()


# ---------------------------------------
# Fix Kaggle → Local Path
# ---------------------------------------
def normalize_path(path):
    if "/kaggle/working/" in path:
        return path.replace("/kaggle/working/", "")
    return path


# ---------------------------------------
# Encode Single Image
# ---------------------------------------
def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        emb = model.get_image_features(**inputs)

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu()


# ---------------------------------------
# Encode Text
# ---------------------------------------
def encode_text(text):
    inputs = processor(text=text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        emb = model.get_text_features(**inputs)

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu()


# ---------------------------------------
# Evaluation Function
# ---------------------------------------
def evaluate(csv_path):

    print("\n==============================")
    print(f"Evaluating: {csv_path}")
    print("==============================")

    df = pd.read_csv(csv_path)

    # Normalize paths
    df["image_path"] = df["image_path"].apply(normalize_path)

    # Unique images
    unique_images = df["image_path"].unique()
    image_to_index = {img: idx for idx, img in enumerate(unique_images)}

    # -----------------------------
    # Encode All Images Once
    # -----------------------------
    print("Encoding images...")
    image_embeddings = []

    for img_path in tqdm(unique_images):
        image_embeddings.append(encode_image(img_path))

    image_embeddings = torch.cat(image_embeddings)  # [N_images, 512]

    # -----------------------------
    # Metrics
    # -----------------------------
    r1 = 0
    r5 = 0
    r10 = 0
    mrr = 0

    print("Evaluating captions...")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        caption = row["caption"]
        true_image = row["image_path"]
        true_index = image_to_index[true_image]

        text_emb = encode_text(caption)  # [1, 512]

        sims = (text_emb @ image_embeddings.T).squeeze(0)
        ranked_indices = torch.argsort(sims, descending=True)

        rank_position = (ranked_indices == true_index).nonzero(as_tuple=True)[0].item() + 1

        if rank_position <= 1:
            r1 += 1
        if rank_position <= 5:
            r5 += 1
        if rank_position <= 10:
            r10 += 1

        mrr += 1.0 / rank_position

    total = len(df)

    print("\nResults:")
    print(f"Total Queries : {total}")
    print(f"Recall@1      : {r1/total:.4f}")
    print(f"Recall@5      : {r5/total:.4f}")
    print(f"Recall@10     : {r10/total:.4f}")
    print(f"MRR           : {mrr/total:.4f}")
    print("==============================\n")


# ---------------------------------------
# Run
# ---------------------------------------
if __name__ == "__main__":

    evaluate("val_clip.csv")
    evaluate("test_clip.csv")
