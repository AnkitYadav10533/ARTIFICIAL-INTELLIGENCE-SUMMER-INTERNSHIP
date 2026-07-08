# Male & Female Image Classifier

A simple, interactive machine learning web application built using **Streamlit** and **Scikit-learn** to classify images into **Male** or **Female** categories. The application uses a trained Logistic Regression model (`male_female_model.pkl`) to make predictions on uploaded images and display the prediction confidence.

---

## 📂 Project Structure

```text
Male_Female_Classifier/
├── app (1).py            # Streamlit web application
├── train_model_py.py     # Python script to train the Logistic Regression model
├── male_female_model.pkl # Trained Scikit-learn model file (serialized)
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation (this file)
```

---

## 🛠️ Installation & Setup

Ensure you have Python 3.8+ installed. Follow these steps to set up and run the project:

### 1. Install Dependencies
Install all the required python libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Run the Application
Start the Streamlit development server by running:

```bash
streamlit run "app (1).py"
```

Once running, the application will automatically open in your default browser at `http://localhost:8501`.

---

## 🧠 Model Training

The classifier model is trained on preprocessed image data (flattened $64 \times 64$ RGB pixels) using Logistic Regression. 

To retrain or train the model again:
1. Ensure your dataset is placed at the path specified in `train_model_py.py` (by default `/content/sample_data/dataset` containing `male` and `female` subfolders).
2. Run the training script:
   ```bash
   python train_model_py.py
   ```
This will save a new `male_female_model.pkl` in your workspace.

---

> [!NOTE]
> **Label Mismatch Bug in Streamlit App**
>
> In the original `app (1).py` code, the prediction display outputs:
> - `0` $\rightarrow$ **🐱 Prediction: CAT** (Instead of **Male**)
> - `1` $\rightarrow$ **🐶 Prediction: DOG** (Instead of **Female**)
>
> To fix this and display correct labels, you can update lines 67-76 in `app (1).py` to match the `["male", "female"]` classes defined in `train_model_py.py`.
