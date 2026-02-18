Semantic Visual Retrieval Engine
================================

**Region-Aware Semantic Image Retrieval using Fine-Tuned CLIP and FAISS**

Overview
--------

This project implements a **Semantic Visual Retrieval Engine** for satellite imagery using a fine-tuned CLIP model and FAISS-based vector indexing.

The system supports:

*   Text-to-image semantic retrieval
    
*   Region-level visual evidence highlighting
    
*   Efficient vector similarity search using FAISS
    
*   Adaptive similarity ranking
    
*   Quantitative retrieval evaluation
    

Unlike traditional classifiers, this system performs **embedding-based semantic ranking** and highlights the most relevant regions inside an image using similarity scoring.

Dataset
-------

RSICD – Remote Sensing Image Caption Dataset

Kaggle Dataset:[https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset](https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset)

The dataset contains:

*   Satellite imagery
    
*   Human-written scene captions
    
*   Multiple semantic concepts per image
    

Images are not included in this repository due to size constraints.Please download from Kaggle and place them into:

```
images/      
    train/      
    val/      
    test/   
```

Model Fine-Tuning
-----------------

The CLIP model was fine-tuned on RSICD using contrastive learning.

Fine-tuning notebook: [https://www.kaggle.com/code/vjx021/clip-finetune](https://www.kaggle.com/code/vjx021/clip-finetune)

### Training Strategy

*   Base Model: OpenAI CLIP (ViT-based)
    
*   Fine-tuned on image–caption pairs
    
*   L2-normalized embeddings
    
*   Contrastive cross-entropy loss
    
*   Cosine similarity optimization
    

The model learns to:

*   Increase similarity between matching image-caption pairs
    
*   Decrease similarity for non-matching pairs
    

System Architecture
-------------------

### Text → Image Retrieval

```   
Text Query
     ↓  
CLIP Text Encoder     
     ↓  
Vector Embedding (512-d)
     ↓
FAISS Index Search
     ↓
Top-K Ranked Images   
```

### Region-Level Visual Retrieval

```
Selected Image
     ↓
Grid-Based Region Split
     ↓
CLIP Region Encoding
     ↓
Similarity Ranking
     ↓
Bounding Box Evidence Highlighting   
```

Application Demo
----------------

### 1️⃣ Text → Image Retrieval

Semantic image search using FAISS-based nearest neighbor search over fine-tuned CLIP embeddings.

![App Screenshot](https://github.com/Vijay2101/Semantic-Visual-Retrieval-Engine/blob/main/assets/text_to_image.png?raw=true)

Images are ranked using cosine similarity between normalized text and image embeddings.

### 2️⃣ Region-Level Visual Retrieval (Visual RAG)

The selected image is spatially divided into regions.Each region is encoded independently and ranked against the query embedding.

#### Input Interface

![App Screenshot](https://github.com/Vijay2101/Semantic-Visual-Retrieval-Engine/blob/main/assets/visual_rag_input.png?raw=true)

#### Region-Level Evidence Output

![App Screenshot](https://github.com/Vijay2101/Semantic-Visual-Retrieval-Engine/blob/main/assets/visual_rag_output.png?raw=true)


The system:

*   Highlights top semantic regions
    
*   Uses adaptive similarity normalization
    
*   Avoids brittle fixed thresholds
    
*   Displays ranked region similarity scores
    

Technical Details
-----------------

### Embeddings

*   512-dimensional CLIP embeddings
    
*   L2 normalization applied
    
*   Cosine similarity via dot product
    

### Vector Database

*   FAISS for efficient nearest-neighbor search
    
*   Precomputed image embeddings
    
*   Fast top-K retrieval
    

### Adaptive Similarity Strategy

Instead of fixed similarity thresholds:

*   Always select Top-4 regions
    
*   Additionally select regions within 95% of maximum similarity
    
*   Maintains ranking stability
    
*   Reduces noisy detections
    

Evaluation
----------

Retrieval performance evaluated on the **test split**.

### Best Test Performance

*   Recall@1: 10.8%
    
*   Recall@5: 31.1%
    
*   Recall@10: 47.5%
    
*   MRR: 0.2211
    

The model retrieves the correct image within the top 10 results nearly **48% of the time**, demonstrating strong semantic alignment between text queries and satellite imagery after domain adaptation.

To reproduce evaluation:

```
   python evaluate_retrieval.py  
 ```

Metrics computed:

*   Recall@K
    
*   Mean Reciprocal Rank (MRR)
    

Installation
------------

### 1️⃣ Clone Repository

```   
git clone https://github.com/your-username/semantic-visual-retrieval.git  cd semantic-visual-retrieval   
```

### 2️⃣ Install Dependencies

```   
pip install -r requirements.txt   
```

### 3️⃣ Download Dataset

Download from Kaggle:

[https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset](https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset)

Place inside:

```   
images/train  
images/val  
images/test   
```

### 4️⃣ Build FAISS Index

```   
python build_faiss_index.py   
```

### 5️⃣ Run Application

```   
streamlit run app.py   
```

Repository Structure
--------------------

```   
app.py  
build_faiss_index.py  
evaluate_retrieval.py  
clip-vit-b16-finetune.ipynb  
requirements.txt  
README.md  
assets/  
images/   (sample images only)  
clip-rsicd  (clip model-download it from the kaggle output link or finetune using the ipynb file)   
```

Key Contributions
-----------------

*   Domain adaptation of CLIP for satellite imagery
    
*   FAISS-based scalable semantic search
    
*   Region-aware evidence extraction
    
*   Adaptive similarity normalization
    
*   End-to-end evaluation pipeline
    

