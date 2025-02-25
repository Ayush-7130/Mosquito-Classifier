import tensorflow as tf
import numpy as np
import streamlit as st
import os
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import preprocess_input

# Define the correct model architecture
def build_model():
    # Data Augmentation (Keep same as training)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),                     # Randomly flips images horizontally (Useful for object recognition where left/right orientation doesn't matter)
        layers.RandomRotation(0.2),                          # Rotates images by up to 20% (0.2 * 360° = 72° max)
        layers.RandomHeight(0.2),                            # Adjusts image height by ±20% (Helps in handling different aspect ratios.)
        layers.RandomWidth(0.2),                             # Adjusts image width by ±20% (Useful when objects may appear stretched or compressed.)
        layers.RandomZoom(0.2),                              # Randomly zooms images by ±20%
        # preprocessing.Rescaling(1/255.)                    # rescale inputs of images to between 0 & 1, required for models like ResNet50 but i am using EfficientNetX
    ], name="data_augmentation")

    # Setup the base model and freeze its layers (this will extract features)
    base_model = tf.keras.applications.EfficientNetB4(include_top=False)
    base_model.trainable = True  # Unfreeze for fine-tuning

    for layer in base_model.layers[:200]:  # Keep first 200 layers frozen
        layer.trainable = False

    # Setup model architecture with trainable top layers
    inputs = layers.Input(shape=(224, 224, 3), name="input_layer")
    x = data_augmentation(inputs)                                                                # augment images (only happens during training phase)
    x = base_model(x, training=False)                                                            # put the base model in inference mode so weights which need to stay frozen, stay frozen
    x = layers.GlobalAveragePooling2D(name="global_avg_pool_layer")(x)                           # Converts feature maps into a 1D vector. Reduces parameters and prevents overfitting.
    outputs = layers.Dense(3, activation="softmax", name="output_layer")(x)                      # Final output layer with 3 classes. Uses softmax activation for multi-class classification.
    model = tf.keras.Model(inputs, outputs)
    
    return model

def find_latest_weights(directory="Checkpoint"):
    """Finds the latest weights file in the directory."""
    weight_files = [f for f in os.listdir(directory) if f.endswith(".weights.h5")]
    if not weight_files:
        st.error("No saved weights found! Train the model first.")
        return None
    latest_file = max(weight_files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return os.path.join(directory, latest_file)

@st.cache_resource
def load_model():
    """Loads model and applies latest weights."""
    model = build_model()
    latest_weights = find_latest_weights()
    if latest_weights:
        model.load_weights("Checkpoint/latest.weights.h5")
        return model
    return None


def preprocess_image(image):
    """Prepares the image for model prediction using EfficientNetB4 preprocessing."""
    image = image.resize((224, 224))  
    image = np.array(image)  
    image = preprocess_input(image)  # Use EfficientNet's preprocessing
    image = np.expand_dims(image, axis=0)  
    return image


CLASS_NAMES = [
    "Aedes Aegypti", "Anopheles Stephensi", "Culex Quinquefasciatus"
]

# Streamlit UI
st.title("Mosquito Species Classifier 🦟")
st.write("Upload an image to classify it among the 3 species.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    if model is not None:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        st.write("Raw Model Output:", prediction)  # Debugging step
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.success(f"### Prediction: {predicted_class} ({confidence*100:.2f}% confidence)")
    else:
        st.error("Model could not be loaded. Check training and saving process!")
