import os
import json
import torch
import faiss
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from dotenv import load_dotenv
from groq import Groq
from transformers import CLIPModel, CLIPProcessor


# =====================================================
# CONFIG
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("clip-rsicd").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("clip-rsicd")
    model.eval()
    return model, processor


@st.cache_resource
def load_faiss():
    index = faiss.read_index("faiss.index")
    with open("image_paths.pkl", "rb") as f:
        image_paths = pickle.load(f)
    return index, image_paths


@st.cache_resource
def load_groq():
    return Groq(api_key=GROQ_API_KEY)


clip_model, clip_processor = load_clip()
faiss_index, image_paths = load_faiss()
groq_client = load_groq()


# =====================================================
# ENCODERS
# =====================================================
def encode_text(text):
    inputs = clip_processor(text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


def encode_image(image):
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


# =====================================================
# REGION SPLITTING
# =====================================================
def split_into_regions(image, grid_size=4):
    w, h = image.size
    regions, boxes = [], []
    dw, dh = w // grid_size, h // grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            left = j * dw
            top = i * dh
            right = (j + 1) * dw
            bottom = (i + 1) * dh

            regions.append(image.crop((left, top, right, bottom)))
            boxes.append((left, top, right, bottom))

    return regions, boxes


# =====================================================
# GROQ JSON EXPLANATION
# =====================================================
def groq_json_reasoning(query, evidence):

    prompt = f"""
Return ONLY valid JSON.

User Query:
{query}

Retrieved Visual Evidence:
{json.dumps(evidence, indent=2)}

Format:
{{
  "answer": "...",
  "reasoning": "...",
  "regions_used": [1,2,...]
}}

No extra text.
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a grounded visual reasoning engine."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )

    return json.loads(response.choices[0].message.content)


# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(layout="wide")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("Visual Retrieval System")
    st.markdown("### Model Info")
    st.write("Dataset: RSICD")
    st.write("Model: Fine-tuned CLIP")
    st.write("Vector DB: FAISS")
    st.write("Device:", DEVICE)

    st.markdown("---")
    mode = st.radio(
        "Select Mode",
        ["Text → Image Retrieval", "Visual RAG (Region-Level)"]
    )

# ---------- MAIN HEADER ----------
st.markdown(
    "<h1 style='text-align: center;'>Semantic Visual Retrieval & Region-Level Evidence</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: gray;'>Fine-tuned CLIP + FAISS-based Retrieval</p>",
    unsafe_allow_html=True
)

st.markdown("---")


# =====================================================
# MODE 1 — TEXT TO IMAGE
# =====================================================
if mode == "Text → Image Retrieval":

    col_left, col_center, col_right = st.columns([1,2,1])

    with col_center:
        query = st.text_input("Enter semantic query", "airport with runway")
        search_button = st.button("Search")

    if search_button:

        query_emb = encode_text(query)
        D, I = faiss_index.search(query_emb, k=6)

        st.markdown("### Top Retrieved Images")

        result_cols = st.columns(3)

        for idx, score, i in zip(I[0], D[0], range(len(I[0]))):
            with result_cols[i % 3]:
                st.image(image_paths[idx], width=280)
                st.caption(f"Similarity: {score:.4f}")


# =====================================================
# MODE 2 — VISUAL RAG
# =====================================================
if mode == "Visual RAG (Region-Level)":

    col_left, col_center, col_right = st.columns([1,2,1])

    with col_center:
        selected_image = st.selectbox("Select image", image_paths)
        image = Image.open(selected_image).convert("RGB")
        st.image(image, width=550)

        query = st.text_input("Ask about this image", "airport terminal buildings")
        run_button = st.button("Analyze Regions")

    if run_button:

        regions, boxes = split_into_regions(image)

        region_embs = []
        for region in regions:
            region_embs.append(encode_image(region))

        region_embs = np.vstack(region_embs)
        query_emb = encode_text(query)

        sims = np.dot(query_emb, region_embs.T)[0]

        # ----- Ranking Strategy (unchanged) -----
        top4_indices = sims.argsort()[-4:][::-1]

        max_score = sims.max()
        relative_threshold = 0.90 * max_score
        relative_indices = np.where(sims >= relative_threshold)[0]

        combined_indices = list(set(top4_indices.tolist() + relative_indices.tolist()))
        combined_indices = sorted(
            combined_indices,
            key=lambda x: sims[x],
            reverse=True
        )
        combined_indices = combined_indices[:8]

        selected_boxes = [boxes[i] for i in combined_indices]
        selected_scores = [float(sims[i]) for i in combined_indices]

        if max_score < 0.15:
            st.warning("Weak semantic match detected.")
            st.info(f"Max similarity score: {max_score:.4f}")

        # ---------- DRAW IMAGE WITH BOXES ----------
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(image)
        ax.axis("off")

        for box in selected_boxes:
            l, t, r, b = box
            ax.add_patch(
                plt.Rectangle(
                    (l, t),
                    r - l,
                    b - t,
                    fill=False,
                    edgecolor="red",
                    linewidth=3
                )
            )

        col_left2, col_center2, col_right2 = st.columns([1,2,1])
        with col_center2:
            st.pyplot(fig)

        st.markdown("### Retrieved Regions")

        score_cols = st.columns(2)

        for i, (idx, score) in enumerate(zip(combined_indices, selected_scores)):
            with score_cols[i % 2]:
                st.markdown(
                    f"""
                    **Rank {i+1}**  
                    Region ID: {idx+1}  
                    Similarity: `{score:.4f}`
                    """
                )

        st.success(
            f"Max similarity: {max_score:.4f} | Dynamic threshold: {relative_threshold:.4f}"
        )
