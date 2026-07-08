import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------
# 1. Page Configuration & Aesthetic Theme
# ----------------------------------------------------
st.set_page_config(
    page_title="K-Means Clustering App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injecting Custom CSS for a beautiful modern design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Outfit:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, .title-text {
    font-family: 'Outfit', sans-serif;
}

/* Custom Gradient Button style */
.stButton>button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    width: 100%;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    color: white;
    border: none;
}

/* Card Style for stats */
.metric-card {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    text-align: center;
}

.metric-card h3 {
    margin: 0;
    color: #6c757d;
    font-size: 0.9rem;
    text-transform: uppercase;
}

.metric-card p {
    margin: 5px 0 0 0;
    font-size: 1.8rem;
    font-weight: 700;
    color: #343a40;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Data Loading and Helper Functions
# ----------------------------------------------------
@st.cache_data
def load_data(filepath="income.csv"):
    """
    Safely load the CSV file using a relative path.
    """
    try:
        df = pd.read_csv(filepath)
        # Strip whitespaces from column names just in case
        df.columns = [col.strip() for col in df.columns]
        return df
    except FileNotFoundError:
        st.error(f"Error: The dataset file at '{filepath}' was not found. Please place it in the same directory.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the dataset: {e}")
        return None

def scale_features(df, features):
    """
    Scale the selected features to the range [0, 1] using MinMaxScaler.
    """
    df_scaled = df.copy()
    scaler = MinMaxScaler()
    for col in features:
        df_scaled[col] = scaler.fit_transform(df[[col]])
    return df_scaled, scaler

def run_kmeans(df_scaled, features, n_clusters):
    """
    Run KMeans clustering on scaled features and return the fitted model and predictions.
    """
    km = KMeans(n_clusters=n_clusters, random_state=42)
    y_predicted = km.fit_predict(df_scaled[features])
    return km, y_predicted

# ----------------------------------------------------
# 3. Main Dashboard Application
# ----------------------------------------------------
def main():
    # Styled Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 12px; margin-bottom: 25px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h1 style="margin: 0; font-weight: 700; font-size: 2.6rem; letter-spacing: -0.5px;">K-Means Clustering App</h1>
        <p style="margin: 8px 0 0 0; font-weight: 300; font-size: 1.1rem; opacity: 0.9;">Analyze, segment, and visualize customer demographic data interactively</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    df = load_data("income.csv")
    if df is None:
        return

    # ----------------------------------------------------
    # 4. Sidebar Control Panel
    # ----------------------------------------------------
    st.sidebar.markdown("### ⚙️ Cluster Configuration")
    
    # Identify numerical columns for clustering
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Feature Selection (defaulting to Age and Income($) if present)
    default_features = [col for col in ["Age", "Income($)"] if col in numerical_cols]
    if len(default_features) < 2 and len(numerical_cols) >= 2:
        default_features = numerical_cols[:2]
        
    selected_features = st.sidebar.multiselect(
        "Select features for clustering:",
        options=numerical_cols,
        default=default_features
    )
    
    # K parameter selector
    k_value = st.sidebar.slider(
        "Number of Clusters (K):",
        min_value=2,
        max_value=10,
        value=3,
        step=1
    )
    
    # Trigger clustering button
    run_btn = st.sidebar.button("Run Clustering")

    # If features are not sufficiently selected
    if len(selected_features) < 2:
        st.warning("Please select at least 2 numerical features in the sidebar to run the K-Means clustering.")
        return

    # Check for K-Means execution trigger/state
    if "kmeans_ran" not in st.session_state:
        st.session_state["kmeans_ran"] = False
        st.session_state["clustered_df"] = None
        st.session_state["centroids"] = None
        st.session_state["k_val"] = k_value

    # Run clustering if button is pressed or if state needs initialization
    if run_btn or not st.session_state["kmeans_ran"]:
        df_scaled, scaler = scale_features(df, selected_features)
        km_model, y_pred = run_kmeans(df_scaled, selected_features, k_value)
        
        # Add cluster labels to the original dataframe
        df_clustered = df.copy()
        df_clustered['cluster'] = y_pred
        
        # Save results in session state
        st.session_state["kmeans_ran"] = True
        st.session_state["clustered_df"] = df_clustered
        st.session_state["scaled_df"] = df_scaled
        st.session_state["centroids"] = km_model.cluster_centers_
        st.session_state["k_val"] = k_value
        st.session_state["selected_features"] = selected_features

    # ----------------------------------------------------
    # 5. Main Dashboard Layout - Tabs
    # ----------------------------------------------------
    tab_preview, tab_analysis, tab_elbow = st.tabs([
        "📁 Dataset Insights", 
        "📊 Clustering Analysis", 
        "📈 Elbow Method Analysis"
    ])

    # ------------------- TAB 1: DATASET PREVIEW & INFO -------------------
    with tab_preview:
        st.subheader("Dataset Details & Summary Statistics")
        
        # Metric cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>Total Records</h3><p>{df.shape[0]}</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>Dataset Columns</h3><p>{df.shape[1]}</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>Missing Values</h3><p>{df.isnull().sum().sum()}</p></div>', unsafe_allow_html=True)

        st.write("---")

        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown("#### Raw Dataset Preview")
            st.dataframe(df, use_container_width=True)
            
        with col_right:
            st.markdown("#### Missing Value Check")
            null_summary = df.isnull().sum().to_frame(name="Missing Counts")
            st.table(null_summary)
            
            st.markdown("#### Summary Statistics")
            st.dataframe(df.describe().T[["mean", "std", "min", "max"]], use_container_width=True)

    # ------------------- TAB 2: CLUSTERING ANALYSIS -------------------
    with tab_analysis:
        st.subheader(f"K-Means Clustering Results (K={st.session_state['k_val']})")
        
        df_clustered = st.session_state["clustered_df"]
        df_scaled = st.session_state["scaled_df"]
        centroids = st.session_state["centroids"]
        curr_features = st.session_state["selected_features"]
        curr_k = st.session_state["k_val"]
        
        col_plot, col_table = st.columns([3, 2])
        
        with col_plot:
            st.markdown(f"#### Scatter Plot of Clusters ({curr_features[0]} vs. {curr_features[1]})")
            st.caption("Note: Features are normalized to [0,1] range to match notebook scaling logic.")
            
            # Map dynamic colors (excluding purple for centroids)
            color_palette = ['green', 'red', 'black', 'blue', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            for i in range(curr_k):
                cluster_subset = df_scaled[df_clustered['cluster'] == i]
                c = color_palette[i % len(color_palette)]
                ax.scatter(
                    cluster_subset[curr_features[0]], 
                    cluster_subset[curr_features[1]], 
                    color=c, 
                    label=f'Cluster {i}',
                    s=60,
                    alpha=0.8
                )
                
            # Centroids as purple stars
            ax.scatter(
                centroids[:, 0], 
                centroids[:, 1], 
                color='purple', 
                marker='*', 
                s=250, 
                label='centroid'
            )
            
            ax.set_xlabel(f'Scaled {curr_features[0]}')
            ax.set_ylabel(f'Scaled {curr_features[1]}')
            ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#e9ecef')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Rendering plot
            st.pyplot(fig)
            
        with col_table:
            st.markdown("#### Clustered Output Data Table")
            st.dataframe(df_clustered, use_container_width=True)
            
            # Download clustered dataset
            csv = df_clustered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Clustered Dataset (CSV)",
                data=csv,
                file_name="clustered_income_data.csv",
                mime="text/csv"
            )
            
            st.markdown("#### Scaled Cluster Centroids")
            # Create a dataframe showing centroid coordinates
            centroid_df = pd.DataFrame(centroids, columns=[f"Scaled {f}" for f in curr_features])
            centroid_df.index.name = "Cluster Label"
            st.dataframe(centroid_df, use_container_width=True)

    # ------------------- TAB 3: ELBOW METHOD ANALYSIS -------------------
    with tab_elbow:
        st.subheader("Elbow Curve Optimization")
        st.markdown("The Elbow Method computes the sum of squared errors (SSE) for different values of K to help identify the optimal number of clusters.")
        
        # Calculate SSE for k=1 to 9
        sse = []
        k_range = range(1, 10)
        for k in k_range:
            km_temp = KMeans(n_clusters=k, random_state=42)
            km_temp.fit(df_scaled[curr_features])
            sse.append(km_temp.inertia_)
            
        fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
        ax_elbow.plot(k_range, sse, marker='o', color='#764ba2', linewidth=2, markersize=6)
        ax_elbow.set_xlabel('K (Number of Clusters)')
        ax_elbow.set_ylabel('Sum of Squared Error (SSE)')
        ax_elbow.set_title('Elbow Curve to Find Optimal K')
        ax_elbow.grid(True, linestyle='--', alpha=0.5)
        
        col_elbow_plot, col_elbow_desc = st.columns([3, 2])
        with col_elbow_plot:
            st.pyplot(fig_elbow)
        with col_elbow_desc:
            st.markdown("#### How to interpret this curve:")
            st.markdown("""
            - The **SSE** decreases as the number of clusters (K) increases.
            - Look for the 'elbow' point where the rate of decrease shifts from rapid to slow.
            - In the original dataset, the elbow point is clearly at **K = 3**, verifying that 3 clusters is the optimal partition.
            """)

if __name__ == "__main__":
    main()
