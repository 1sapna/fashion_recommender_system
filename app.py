import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from PIL import Image
import os

# Load feature list and filenames
feature_list = np.array(pickle.load(open('Images_features.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Modify filenames to remove the initial part of the path
filenames = [os.path.join('images', os.path.basename(f)) for f in filenames]

# Load ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Create Sequential model with GlobalMaxPooling2D layer
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Streamlit UI
st.title('👗 Fashion Recommender System')
st.write("Welcome to the Fashion Recommender System! Upload an image to find similar fashion items.")

# Function for feature extraction
def feature_extraction(img_file, model):
    img = Image.open(img_file).convert('RGB').resize((224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    with st.spinner('Extracting features...'):
        result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Function for recommendation
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File upload and processing
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    display_image = Image.open(uploaded_file)
    st.image(display_image, caption='Uploaded Image', use_column_width=True)
    
    # Extract features from uploaded image
    features = feature_extraction(uploaded_file, model)
    
    # Perform recommendation based on extracted features
    indices = recommend(features, feature_list)
    
    # Initialize the list to store original paths
    original_paths = []

    # Create columns for displaying images
    st.write("### Similar Items You Might Like:")
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            original_paths.append(filenames[indices[0][i + 1]])
    
    # Define the prefix to remove
    prefix = 'kaggle/input/fashion-product-images-small/images/'
    
    # Remove the prefix and create new paths
    updated_paths = [path.replace(prefix, 'fr/images') for path in original_paths]
    
    # Print the updated paths to verify
    st.write("Here are some fashion items similar to the one you uploaded:")
    
    # Display the updated images with captions
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(updated_paths[i], caption=f'Similar Item {i+1}', use_column_width=True)

st.sidebar.title("About")
st.sidebar.info("""
This app uses a pre-trained ResNet50 model to find and recommend similar fashion items based on the uploaded image. 
Developed with 💖 by [Sapna].
""")

st.sidebar.title("How to Use")
st.sidebar.write("""
1. Upload an image of a fashion item.
2. Wait for the system to find similar items.
3. View the recommended items on the main page.
""")
