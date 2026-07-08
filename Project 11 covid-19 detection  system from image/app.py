import streamlit as st
import os
import glob

# Set page config at the very beginning
st.set_page_config(
    page_title="COVID-19 Chest X-Ray Detector",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Global Styles */
    .stApp {
        background-color: #0d1117;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Header Styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
    }
    
    .title-gradient {
        background: linear-gradient(45deg, #ff4b4b, #ff7675, #6c5ce7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #8b949e;
        font-size: 1.15rem;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto 1.5rem auto;
    }

    /* Cards */
    .glass-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
    }
    
    /* Result Containers */
    .result-container {
        text-align: center;
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
    }
    
    .badge-covid {
        background-color: rgba(231, 76, 60, 0.15);
        border: 2px solid #e74c3c;
        color: #e74c3c;
        font-size: 1.8rem;
        font-weight: 800;
        padding: 10px 20px;
        border-radius: 50px;
        display: inline-block;
        box-shadow: 0 0 15px rgba(231, 76, 60, 0.2);
    }
    
    .badge-normal {
        background-color: rgba(46, 204, 113, 0.15);
        border: 2px solid #2ecc71;
        color: #2ecc71;
        font-size: 1.8rem;
        font-weight: 800;
        padding: 10px 20px;
        border-radius: 50px;
        display: inline-block;
        box-shadow: 0 0 15px rgba(46, 204, 113, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Layout Title
st.markdown("""
<div class="main-header">
    <div class="title-gradient">COVID-19 Chest X-Ray AI Detector</div>
    <div class="subtitle">An intelligent binary classification model to detect COVID-19 anomalies from Chest X-Ray images.</div>
</div>
""", unsafe_allow_html=True)

# Lazy imports for TensorFlow to improve initial load time
@st.cache_resource
def get_tf_and_model_helpers():
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    return tf, Image, np

# Automatic detection function
def detect_model_files():
    # Scan current folder for any files matching *.h5 or *.keras
    models = glob.glob("*.h5") + glob.glob("*.keras")
    # Return unique, sorted file names
    return sorted(list(set(models)))

# Session state initialization
if "use_random_model" not in st.session_state:
    st.session_state.use_random_model = False

# Sidebar Info and Model Management
with st.sidebar:
    st.markdown("### 🩺 Model Configuration")
    
    detected_models = detect_model_files()
    selected_model_path = None
    
    if detected_models:
        if len(detected_models) == 1:
            selected_model_path = detected_models[0]
            st.success(f"✅ Detected Model: `{selected_model_path}`")
        else:
            selected_model_path = st.selectbox("📂 Select Active Model", detected_models)
            st.success(f"✅ Selected Model: `{selected_model_path}`")
            
        if st.button("Delete active model weights file"):
            try:
                os.remove(selected_model_path)
                st.session_state.use_random_model = False
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting file: {e}")
    else:
        st.warning("⚠️ No model file (.h5 or .keras) detected in workspace.")
        
        # Uploading model through sidebar
        uploaded_model_file = st.file_uploader("Upload model file directly (.h5 or .keras)", type=["h5", "keras"])
        if uploaded_model_file is not None:
            target_name = uploaded_model_file.name
            with st.spinner("Saving uploaded model..."):
                with open(target_name, "wb") as f:
                    f.write(uploaded_model_file.getbuffer())
            st.success(f"🎉 Saved model as `{target_name}`!")
            st.rerun()
            
        st.divider()
        st.markdown("#### Test/Development Mode")
        if not st.session_state.use_random_model:
            if st.button("Initialize Untrained Model"):
                st.session_state.use_random_model = True
                st.rerun()
        else:
            st.info("💡 Running in Test Mode (using untrained model structure).")
            if st.button("Disable Test Mode"):
                st.session_state.use_random_model = False
                st.rerun()

# Load TF and Model helpers
tf, Image, np = get_tf_and_model_helpers()

@st.cache_resource
def load_keras_model(path, use_random=False):
    if use_random:
        # Create untrained sequential model based on original architecture
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(299, 299, 3)),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
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
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy']
        )
        return model, "Untrained / Randomly Initialized Model"
    else:
        try:
            model = tf.keras.models.load_model(path)
            return model, f"Trained Model ({path})"
        except Exception as e:
            return None, f"Error loading model: {str(e)}"

# Main workflow logic
model = None
model_name = ""

if selected_model_path is not None and not st.session_state.use_random_model:
    with st.spinner(f"Loading {selected_model_path} (this may take a few seconds)..."):
        model, model_name = load_keras_model(selected_model_path, use_random=False)
elif st.session_state.use_random_model:
    with st.spinner("Initializing neural network..."):
        model, model_name = load_keras_model(None, use_random=True)

# Main UI components
if model is None:
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-top:0; color: #ff7675;">🩺 Model Weights Missing</h3>
        <p>This application requires a trained Convolutional Neural Network (CNN) model file (<code>.h5</code> or <code>.keras</code>) to perform image classifications.</p>
        <p>Because model weights are large files, they are typically excluded from Git repositories (via <code>.gitignore</code>) and not pushed to GitHub.</p>
        <h4 style="color: #6c5ce7; margin-top: 20px;">How to resolve this:</h4>
        <ol>
            <li><b>Train the Model:</b> Run the model training script locally to train the model on your dataset:
                <pre style="background-color: #1c2128; padding: 12px; border-radius: 6px; color: #e1e4e8; overflow-x: auto; font-family: monospace;">
python project_15_run_on_colab_detection_of_covid_19_from_chest_x_ray.py</pre>
                This script saves the trained weights as <code>model.h5</code>.
            </li>
            <li><b>Upload the Model:</b> Once you have the trained <code>model.h5</code> or <code>model.keras</code> file, upload it directly below or place it in the project root directory.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload box right on the main page for convenience!
    st.subheader("📤 Upload Model File")
    main_uploaded_model = st.file_uploader("Drag and drop your trained model file here (.h5 or .keras)", type=["h5", "keras"], key="main_model_uploader")
    if main_uploaded_model is not None:
        target_name = main_uploaded_model.name
        with st.spinner(f"Saving `{target_name}` to the workspace..."):
            with open(target_name, "wb") as f:
                f.write(main_uploaded_model.getbuffer())
        st.success(f"🎉 Saved model as `{target_name}`! Reloading app...")
        st.rerun()
        
    st.info("💡 Alternatively, you can click **'Initialize Untrained Model'** in the sidebar to test the interface layout without loading trained weights.")
else:
    # Model loaded successfully
    st.info(f"🧬 Model Active: **{model_name}**")
    
    # Create Cards for upload
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📷 Upload Chest X-Ray Image")
    uploaded_image = st.file_uploader("Upload a jpeg/png chest X-ray image for analysis:", type=["jpeg", "jpg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_image is not None:
        col1, col2 = st.columns([1, 1], gap="medium")
        
        # Display image
        with col1:
            st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
            st.write("**Uploaded X-Ray Image:**")
            pil_image = Image.open(uploaded_image)
            st.image(pil_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Inference
        with col2:
            st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
            st.write("**Model Analysis:**")
            
            with st.spinner("Analyzing image..."):
                # Preprocess image
                # 1. Resize to (299, 299) and convert to RGB (3 channels)
                resized_img = pil_image.convert("RGB").resize((299, 299))
                # 2. Rescale by 1/255.0
                img_array = np.array(resized_img) / 255.0
                # 3. Add batch dimension
                img_array = np.expand_dims(img_array, axis=0)
                
                # Make prediction
                prediction_value = float(model.predict(img_array)[0][0])
                
            # Classify
            threshold = 0.5
            is_covid = prediction_value > threshold
            
            # Show output
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            if is_covid:
                st.markdown('<span class="badge-covid">COVID DETECTED</span>', unsafe_allow_html=True)
                confidence = prediction_value * 100
                st.markdown(f"<p style='margin-top: 15px; font-size: 1.1rem;'>COVID Likelihood: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)
                st.progress(prediction_value)
            else:
                st.markdown('<span class="badge-normal">NORMAL (HEALTHY)</span>', unsafe_allow_html=True)
                confidence = (1.0 - prediction_value) * 100
                st.markdown(f"<p style='margin-top: 15px; font-size: 1.1rem;'>Normal Likelihood: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)
                st.progress(prediction_value)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.divider()
            st.caption(
                "Note: A score close to 1.0 indicates higher classification affinity "
                "for COVID-19, while a score close to 0.0 indicates Normal chest characteristics."
            )
            st.markdown('</div>', unsafe_allow_html=True)

# Architecture Summary
with st.expander("🛠️ Model Architecture & Technical Details"):
    st.markdown("""
    This model is built using a Convolutional Neural Network (CNN) matching the project configuration:
    - **Input Layer**: accepts image size of `(299, 299, 3)`
    - **Conv2D Block 1**: 32 filters, `(3,3)` kernel, ReLU + MaxPooling `(2,2)`
    - **Conv2D Block 2**: 32 filters, `(3,3)` kernel, ReLU + MaxPooling `(2,2)`
    - **Conv2D Block 3**: 64 filters, `(3,3)` kernel, ReLU + MaxPooling `(2,2)`
    - **Conv2D Block 4**: 128 filters, `(3,3)` kernel, ReLU + MaxPooling `(2,2)`
    - **Dense Block**: Flatten -> 128 units (ReLU) -> 1 unit (Sigmoid activation)
    - **Loss Function**: `binary_crossentropy`
    - **Optimizer**: Adam
    """)
