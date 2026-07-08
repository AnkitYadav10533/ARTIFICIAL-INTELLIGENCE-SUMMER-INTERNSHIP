import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set environment variables to reduce TensorFlow log verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Page Configuration
st.set_page_config(
    page_title="Eye Gender Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Main container and theme settings */
    .reportview-container {
        background: #0f172a;
    }
    
    /* Elegant Title Styling */
    .app-title {
        font-family: 'Inter', 'Outfit', sans-serif;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 5px;
    }
    .app-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 30px;
    }
    
    /* Info Card Design */
    .info-card {
        background: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    }
    
    /* Prediction Container */
    .result-container {
        background: #1e293b;
        padding: 25px;
        border-radius: 16px;
        border-left: 6px solid #6366f1;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.3);
        margin-top: 20px;
    }
    .result-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 15px;
    }
    .result-label-male {
        font-size: 2rem;
        font-weight: 800;
        color: #38bdf8; /* Sky blue for male prediction */
        margin: 5px 0;
    }
    .result-label-female {
        font-size: 2rem;
        font-weight: 800;
        color: #f43f5e; /* Rose for female prediction */
        margin: 5px 0;
    }
    .confidence-text {
        font-size: 1.1rem;
        color: #cbd5e1;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #64748b;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to compile/create the CNN model layout
def create_model_structure():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(299, 299, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model

MODEL_FILENAME = 'model.keras'

# Sidebar Navigation and Configuration
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("Configure model options and check status.")

# Check for model status
model_exists = os.path.exists(MODEL_FILENAME)

if model_exists:
    st.sidebar.success("✅ Pre-trained model found (`model.keras`)")
else:
    st.sidebar.warning("⚠️ `model.keras` not found in workspace!")
    
    # Allow the user to initialize a baseline model for testing
    if st.sidebar.button("🔨 Initialize Baseline Model (weights)", help="Creates a baseline CNN model file with random weights to test the Streamlit app pipeline."):
        with st.spinner("Generating baseline CNN model..."):
            dummy_model = create_model_structure()
            # Save the model
            dummy_model.save(MODEL_FILENAME)
            st.sidebar.success("✅ Baseline model created successfully!")
            st.rerun()

# Custom file uploader for model file (Alternative setup)
uploaded_model = st.sidebar.file_uploader("Upload custom .keras/.h5 model", type=["keras", "h5"])
if uploaded_model is not None:
    with open(MODEL_FILENAME, "wb") as f:
        f.write(uploaded_model.getbuffer())
    st.sidebar.success("✅ Custom model uploaded and saved!")
    st.rerun()

# Load the model helper
@st.cache_resource
def load_trained_model():
    if os.path.exists(MODEL_FILENAME):
        try:
            return tf.keras.models.load_model(MODEL_FILENAME)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

# Load the compiled CNN model
model = load_trained_model()

# Header Section
st.markdown('<div class="app-title">👁️ Eye Gender Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Deep Learning CNN model to identify Male and Female Eyes from photographs</div>', unsafe_allow_html=True)

# Main layout split into columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="info-card"><h4>📸 Upload Eye Photograph</h4>Upload a crop/photo of a human eye to predict if it belongs to a Male or Female.</div>', unsafe_allow_html=True)
    
    uploaded_image = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Eye Photograph', use_container_width=True)

with col2:
    st.markdown('<h4>🔮 Prediction & Insights</h4>', unsafe_allow_html=True)
    
    if uploaded_image is not None:
        if model is None:
            st.error("No model is currently loaded. Please use the control panel in the sidebar to load or generate a baseline model.")
        else:
            with st.spinner("Analyzing eye details..."):
                # Preprocess image
                # 1. Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # 2. Resize to (299, 299) as required by the model
                img_resized = image.resize((299, 299))
                
                # 3. Convert to float numpy array and normalize [0, 1]
                img_array = np.array(img_resized) / 255.0
                
                # 4. Expand dimensions for batch size [1, 299, 299, 3]
                img_array = np.expand_dims(img_array, axis=0)
                
                # 5. Predict
                result = model.predict(img_array)
                confidence = float(result[0][0])
                
                # Binary classification details:
                # Class 0: Female Eyes, Class 1: Male Eyes
                threshold = 0.5
                is_male = confidence > threshold
                
                # Calculate class confidence
                class_confidence = confidence if is_male else (1.0 - confidence)
                
                # Output Results in a stylized card
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown('<div class="result-header">Detection Result</div>', unsafe_allow_html=True)
                
                if is_male:
                    st.markdown(f'<div class="result-label-male">♂️ Male Eye</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-label-female">♀️ Female Eye</div>', unsafe_allow_html=True)
                    
                st.markdown(f'<div class="confidence-text">Confidence level: <b>{class_confidence * 100:.2f}%</b></div>', unsafe_allow_html=True)
                
                # Progress Bar to show model output probability
                st.progress(confidence)
                st.caption(f"Raw Sigmoid Output: {confidence:.4f} (Values > 0.5 tend towards Male eyes)")
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Waiting for an image upload in the left panel to begin classification.")

st.markdown("---")

# Expandable Details Section
with st.expander("🔬 View Model Architecture & Details"):
    st.markdown("""
    ### Convolutional Neural Network (CNN) Structure
    The eye classification model consists of the following layers:
    1. **Conv2D Layer 1**: 32 filters, (3x3) kernel size, ReLU activation. Input shape is `(299, 299, 3)`.
    2. **MaxPooling2D Layer 1**: Pool size (2,2) for downsampling.
    3. **Conv2D Layer 2**: 32 filters, (3x3) kernel size, ReLU activation.
    4. **MaxPooling2D Layer 2**: Pool size (2,2).
    5. **Conv2D Layer 3**: 64 filters, (3x3) kernel size, ReLU activation.
    6. **MaxPooling2D Layer 3**: Pool size (2,2).
    7. **Conv2D Layer 4**: 128 filters, (3x3) kernel size, ReLU activation.
    8. **MaxPooling2D Layer 4**: Pool size (2,2).
    9. **Flatten Layer**: Transforms 2D feature maps into a 1D vector.
    10. **Dense Layer 1**: 128 units, ReLU activation.
    11. **Dense Output Layer**: 1 unit, Sigmoid activation (for binary classification: `Female Eyes` [0] vs. `Male Eyes` [1]).
    """)
    
    st.markdown("""
    ### Data Preprocessing & Augmentation
    * **Target Size**: 299x299 pixels (RGB)
    * **Augmentations used during training**: Rotation (20 deg), Horizontal Flip, Shear (0.2), Zoom (0.2), Width/Height Shifts, Rescale (1/255.0).
    """)

# Footer
st.markdown('<div class="footer">Built with Streamlit & TensorFlow • Eye Gender Classification Project</div>', unsafe_allow_html=True)
