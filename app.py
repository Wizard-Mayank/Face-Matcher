import sys
import types

# --- ULTIMATE MONKEY PATCH START ---
import keras

# 1. Fix the basic layer_utils and data_utils locations
import keras.src.utils.layer_utils as layer_utils
import keras.src.utils.data_utils as data_utils

# 2. Create the 'keras.utils.data_utils' module
if 'keras.utils.data_utils' not in sys.modules:
    utils_data_mod = types.ModuleType('keras.utils.data_utils')
    utils_data_mod.get_file = data_utils.get_file
    sys.modules['keras.utils.data_utils'] = utils_data_mod

# 3. Create the 'keras.utils.layer_utils' module (This fixes your current error)
if 'keras.utils.layer_utils' not in sys.modules:
    utils_layer_mod = types.ModuleType('keras.utils.layer_utils')
    utils_layer_mod.get_source_inputs = layer_utils.get_source_inputs
    sys.modules['keras.utils.layer_utils'] = utils_layer_mod

# 4. Create the 'keras.engine.topology' module
if 'keras.engine.topology' not in sys.modules:
    topo_mod = types.ModuleType('keras.engine.topology')
    topo_mod.get_source_inputs = layer_utils.get_source_inputs
    sys.modules['keras.engine.topology'] = topo_mod

# 5. Ensure the top-level keras.utils has what it needs
if not hasattr(keras.utils, 'layer_utils'):
    keras.utils.layer_utils = layer_utils
if not hasattr(keras.utils, 'get_file'):
    keras.utils.get_file = data_utils.get_file

# --- ULTIMATE MONKEY PATCH END ---

import streamlit as st
import os
import cv2
import pickle
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# configuration
st.set_page_config(page_title="Bollywood Lookalike", layout="wide")

# upload directory
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# --- LOAD MODELS & DATA ---
@st.cache_resource
def load_models():
    detector = MTCNN()
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    return detector, model

@st.cache_data
def load_embeddings():
    feature_list = np.array(pickle.load(open('embedding.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
    return feature_list, filenames

detector, model = load_models()
feature_list, filenames = load_embeddings()

# --- HELPER FUNCTIONS ---
def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    # Convert BGR to RGB for MTCNN
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)

    if not results:
        return None

    x, y, width, height = results[0]['box']
    # Handle potential negative indexing from detector
    x, y = max(0, x), max(0, y)
    face = img_rgb[y:y + height, x:x + width]

    # Preprocessing
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image).astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    
    return model.predict(preprocessed_img).flatten()

def recommend(feature_list, features):
    similarity = cosine_similarity(features.reshape(1, -1), feature_list)
    index_pos = np.argmax(similarity)
    return index_pos

# --- UI INTERFACE ---
st.title('Which Bollywood Celebrity Are You?')

uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    # Save the file
    temp_path = os.path.join('uploads', uploaded_image.name)
    with open(temp_path, 'wb') as f:
        f.write(uploaded_image.getbuffer())

    # Display & Process
    display_image = Image.open(uploaded_image)
    
    with st.spinner('Analyzing face...'):
        features = extract_features(temp_path, model, detector)

    if features is None:
        st.error("No face detected! Please try another photo with a clear view of your face.")
    else:
        index_pos = recommend(feature_list, features)
        
        # Get the raw path from pickle
        raw_filename = filenames[index_pos]
        
        # Convert Windows backslashes to Linux forward slashes
        normalized_path = raw_filename.replace('\\', '/')
        
        # Clean up the actor name for the header
        # This takes 'data/salman_khan/img.jpg' -> 'salman khan'
        path_parts = normalized_path.split('/')
        if len(path_parts) >= 2:
            predicted_actor = path_parts[-2].replace('_', ' ')
        else:
            predicted_actor = "Celebrity"

        # Display Results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Your Photo')
            st.image(display_image, width=300)
        with col2:
            st.subheader(f"You look like: {predicted_actor.title()}")
            try:
                # Use the normalized path here
                st.image(normalized_path, width=300)
            except Exception as e:
                st.error(f"Could not load celebrity image at: {normalized_path}")
                st.write("Not uploaded to GitHub!")