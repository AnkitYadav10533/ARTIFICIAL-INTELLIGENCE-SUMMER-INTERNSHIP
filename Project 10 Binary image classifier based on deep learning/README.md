# Gender Classifier CNN Dashboard

A production-ready Streamlit application that uses a deep Convolutional Neural Network (CNN) to predict binary gender (Female/Male) from an uploaded image. It features a premium, user-friendly interface with Plotly indicator charts, scanning animation diagnostics, and prediction history tracking.

The underlying CNN architecture is trained to recognize abstract facial textures and geometries as outlined in the project notebook.

---

## 🧬 Folder Structure

The project has been organized with a clean, deployable structure:

```text
Gender-Classifier/
├── .gitignore               # Excludes python cache, streamlit local files, and environments
├── app.py                   # Main Streamlit web application code
├── gender_classifier.keras  # Pre-compiled CNN model binary
├── README.md                # Project documentation and deployment guide (this file)
├── requirements.txt         # Package dependencies for Render/Local installation
└── assets/                  # Folder for project assets (tracked via .gitkeep)
```

---

## ✨ Features

- **Drag-and-Drop Image Uploader**: Accepts JPG, JPEG, and PNG targets.
- **Image Metadata Diagnostics**: Extracts image dimensions, size, and file type on upload.
- **Interactive Scanning Animation**: Step-by-step diagnostic status progression (loading model, edge extraction, sigmoid calculation) with a progress bar and overlay animations.
- **Prediction Analytics**: Displays classification label, compute latency, Plotly gauges, and confidence percentage.
- **Prediction Archive History**: Log of current session classification runs in a table, with an archive clear command.
- **AI Explanation Drawer**: Interactive section explaining preprocessing steps, convolution, pooling operations, and Sigmoid functions.
- **Developer Card**: Credits card detailing LinkedIn, GitHub, and Portfolio credentials.

---

## 🛠️ Local Setup Instructions

1. **Clone or Download** the project files to your local machine.
2. **Open your terminal** and navigate to the project directory:
   ```bash
   cd "Binary image classifier"
   ```
3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```
5. Open your web browser and go to the local address (typically `http://localhost:8501`).

---

## 💻 GitHub Upload Steps

To push the project to GitHub before deploying on Render:

1. **Initialize Git repository**:
   ```bash
   git init
   ```
2. **Add all files to staging**:
   ```bash
   git add .
   ```
3. **Commit the changes**:
   ```bash
   git commit -m "Initialize Gender Classifier for Render deployment"
   ```
4. **Create default main branch**:
   ```bash
   git branch -M main
   ```
5. **Add remote origin** (replace with your repo URL):
   ```bash
   git remote add origin https://github.com/AnkitYadav10533/project_AI_ML_Generative_AI.git
   ```
6. **Push to main**:
   ```bash
   git push -u origin main
   ```

---

## 🚀 Render Deployment Steps

To deploy this project as a web service on Render:

1. Sign in to your account at [Render.com](https://render.com/).
2. Click **New** in the dashboard and select **Web Service**.
3. Connect your GitHub repository containing the project files.
4. Set the following configuration parameters:
   - **Name**: `gender-classifier-cnn` (or your preferred name)
   - **Language/Runtime**: `Python`
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
5. Click **Deploy Web Service** to trigger the build and deployment.
