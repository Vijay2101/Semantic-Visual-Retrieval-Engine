import os
import torch
import faiss
import pickle
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_ROOT = "images"
INDEX_FILE = "faiss.index"
PATH_FILE = "image_paths.pkl"

print("Using device:", DEVICE)

# LOAD MODEL
model = CLIPModel.from_pretrained("clip-rsicd").to(DEVICE)
processor = CLIPProcessor.from_pretrained("clip-rsicd")
model.eval()

print("Model loaded.")

# COLLECT ALL IMAGE PATHS
image_paths = []

for split in ["train", "val", "test"]:
    split_folder = os.path.join(IMAGE_ROOT, split)
    if os.path.exists(split_folder):
        for f in os.listdir(split_folder):
            if f.lower().endswith(".jpg"):
                image_paths.append(os.path.join(split_folder, f))

print("Total images found:", len(image_paths))

# ENCODER
def encode_image(path):
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        emb = model.get_image_features(**inputs)

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()

# ENCODE ALL IMAGES
embeddings = []

for i, path in enumerate(image_paths):
    print(f"{i+1}/{len(image_paths)} -> {path}")
    emb = encode_image(path)
    embeddings.append(emb)

embeddings = np.vstack(embeddings).astype("float32")

print("Embedding shape:", embeddings.shape)

# BUILD FAISS INDEX
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
index.add(embeddings)

print("FAISS index built.")

# SAVE INDEX + PATHS
faiss.write_index(index, INDEX_FILE)

with open(PATH_FILE, "wb") as f:
    pickle.dump(image_paths, f)

print("Index saved successfully.")
