# 🧥 Fashion Recommendation System

This is an image-based fashion recommendation system that helps users find visually similar clothing items. Users can upload a picture of an outfit, and the system will return the top 5 most similar items from an inventory using computer vision and machine learning.

## 🔍 Project Overview

- **Objective**: Recommend visually similar fashion items based on an uploaded image.
- **Type**: Content-based recommender system.
- **Tech Stack**: Python, TensorFlow/Keras, scikit-learn, Streamlit.

## 🧠 How It Works

1. **Feature Extraction**: Uses a pretrained CNN (ResNet50) for extracting feature embeddings from clothing images.
2. **Similarity Matching**: Computes cosine similarity between uploaded image and dataset images.
3. **Recommendation**: Uses K-Nearest Neighbors to retrieve top 5 similar items.
4. **Frontend**: Simple Streamlit app to upload an image and display recommendations.

---

## 📁 Project Structure

```
fashion-recommender/
├── data/ # Image dataset
├── embeddings/ # Saved embeddings of dataset images
├── app.py # Streamlit web app
├── feature_extractor.py # CNN model and embedding functions
├── helper.py # Utility functions
├── requirements.txt # Python dependencies
└── README.md

```
---

## 🛠️ Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/fashion-recommender.git
cd fashion-recommender
```
2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

```
3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

## 🧩 Model Details

- **Backbone**: ResNet50 (pretrained on ImageNet)
- **Layer Used**: Global Max Pooling layer for fixed-size embeddings
- **Embedding Size**: 2048-dimensional vector per image

## 🧮 Generating Embeddings
Embeddings are saved using pickle or numpy for fast retrieval.

## 🔎 Search & Recommend
Uses KNN (K-Nearest Neighbors) with cosine similarity to find top 5 similar images.

## 🌐 Running the App
```bash
streamlit run app.py
```
Upload an image via the Streamlit UI

See 5 most similar fashion items displayed

## 📸 Sample Output

 ![Home](home.jpg)
 ![Input](input.jpg)
 ![Output](output.jpg)
