import os
import re
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# Page Config
st.set_page_config(
    page_title="Avengers: Endgame Review Finder",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import base64
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSTER_PATH = os.path.join(BASE_DIR, "assets", "poster.jpeg")

poster_b64 = ""
if os.path.exists(POSTER_PATH):
    try:
        with open(POSTER_PATH, "rb") as f:
            poster_b64 = base64.b64encode(f.read()).decode()
    except Exception:
        pass

# Custom Styling (Avengers themed: Deep purple, dark blue, gold accents)
css_style = """
<style>
    /* Custom Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
    }

    /* Main Container Background */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #150d22 0%, #08050e 100%);
        color: #f1f1f1;
    }

    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background-color: #0b0714 !important;
        border-right: 1px solid #3d236b;
    }

    /* Card styling */
    .fact-card {
        background: rgba(30, 20, 48, 0.6);
        border: 1px solid #4a2b85;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(74, 43, 133, 0.2);
    }
    
    .fact-card-title {
        color: #ffc107;
        font-weight: 600;
        font-size: 1.25rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .fact-card-content {
        color: #e0d5f0;
        font-size: 1.1rem;
        line-height: 1.6;
    }

    .rag-card {
        background: rgba(224, 64, 251, 0.08);
        border: 1px solid #e040fb;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(224, 64, 251, 0.15);
    }

    .rag-card-title {
        color: #e040fb;
        font-weight: 600;
        font-size: 1.25rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .rag-card-content {
        color: #ffffff;
        font-size: 1.1rem;
        line-height: 1.6;
    }

    /* Metric or label styling */
    .metric-badge {
        background: #25163d;
        border: 1px solid #7952b3;
        color: #00ffcc;
        padding: 0.25rem 0.6rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Quick Recommendation Buttons */
    button[kind="secondary"] {
        background-color: #1e1136 !important;
        color: #ffc107 !important;
        border: 1px solid #5a30a8 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    button[kind="secondary"]:hover {
        background-color: #ffc107 !important;
        color: #12072b !important;
        border-color: #ffc107 !important;
        box-shadow: 0 0 10px rgba(255, 193, 7, 0.5);
    }

    /* Hero Banner container with background poster */
    .hero-banner-container {
        position: relative;
        width: 100%;
        height: 380px;
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(74, 43, 133, 0.4);
        opacity: 0;
        animation: fadeIn 1.2s ease-out forwards;
    }

    .hero-banner-bg {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: #150d22;
        background-image: linear-gradient(135deg, #1b003a 0%, #000a21 100%);
        background-size: cover;
        background-position: center 25%;
        filter: brightness(0.75) blur(3px);
        z-index: 1;
        transform: scale(1.02); /* Avoid edge artifacts from blur */
    }

    .hero-banner-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(to bottom, rgba(0, 0, 0, 0.85) 0%, rgba(8, 5, 14, 0.4) 60%, #150d22 100%);
        z-index: 2;
    }

    /* Glassmorphism Title Section */
    .hero-banner-content {
        position: relative;
        z-index: 3;
        text-align: center;
        padding: 2.5rem;
        max-width: 800px;
        background: rgba(21, 13, 34, 0.55);
        backdrop-filter: blur(12px) saturate(160%);
        -webkit-backdrop-filter: blur(12px) saturate(160%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        margin: 0 1.5rem;
    }

    .hero-banner-title {
        color: #ffc107;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0 0 0.75rem 0;
        text-shadow: 0 0 20px rgba(255, 193, 7, 0.4);
        font-family: 'Outfit', sans-serif;
        letter-spacing: 0.5px;
    }

    .hero-banner-subtitle {
        color: #e2d9f3;
        font-size: 1.2rem;
        margin: 0;
        line-height: 1.5;
        font-family: 'Roboto', sans-serif;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Mobile responsive */
    @media (max-width: 768px) {
        .hero-banner-container {
            height: 320px;
        }
        .hero-banner-title {
            font-size: 2.1rem;
        }
        .hero-banner-subtitle {
            font-size: 1rem;
        }
        .hero-banner-content {
            padding: 1.75rem;
        }
    }
</style>
"""

if poster_b64:
    # Inject background image CSS dynamically
    css_style = css_style.replace(
        "background-image: linear-gradient(135deg, #1b003a 0%, #000a21 100%);",
        f'background-image: url("data:image/jpeg;base64,{poster_b64}");'
    )

st.markdown(css_style, unsafe_allow_html=True)


# Helper Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "avengers_endgame_data.csv")
TRANSLATIONS_PATH = os.path.join(BASE_DIR, "reference_translations.txt")

@st.cache_data
def load_data():
    """Loads CSV and reference translations, returning data structures."""
    # 1. Load CSV data
    if not os.path.exists(CSV_PATH):
        st.error(f"Data file not found at {CSV_PATH}. Please make sure it is in the repository.")
        return pd.DataFrame(), {}
    df = pd.read_csv(CSV_PATH)
    
    # 2. Parse Reference Translations
    translations = {}
    if os.path.exists(TRANSLATIONS_PATH):
        with open(TRANSLATIONS_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines to separate blocks
        blocks = re.split(r'\n\s*\n', content)
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            # Check for colon separators (Topic: Content)
            if ':' not in block:
                continue
                
            parts = block.split(':', 1)
            topic = parts[0].strip()
            
            # Skip documentation headers
            if topic.startswith("Reference Translations") or topic.startswith("Format"):
                continue
                
            phrasings_text = parts[1].replace('\n', ' ')
            # Split phrases by ' - ' delimiter
            phrases = [p.strip() for p in phrasings_text.split(' - ') if p.strip()]
            
            if topic and phrases:
                translations[topic] = phrases
    else:
        st.warning(f"Reference translations file not found at {TRANSLATIONS_PATH}.")
        
    return df, translations

df, translations = load_data()

# Keyword mapping for quick direct routing
KEYWORD_MAPPINGS = {
    "director": "Directors",
    "directors": "Directors",
    "directed": "Directors",
    "cast": "Main Cast",
    "actor": "Main Cast",
    "actors": "Main Cast",
    "villain": "Main Villain",
    "bad guy": "Main Villain",
    "thanos": "Main Villain",
    "writer": "Writers",
    "writers": "Writers",
    "wrote": "Writers",
    "runtime": "Runtime",
    "long": "Runtime",
    "length": "Runtime",
    "duration": "Runtime",
    "release": "Release Date",
    "when": "Release Date",
    "date": "Release Date",
    "genre": "Genre",
    "type": "Genre",
    "language": "Language",
    "country": "Country",
    "producer": "Producer",
    "producers": "Producer",
    "produced": "Producer",
    "studio": "Production Company",
    "production": "Production Company",
    "distributor": "Distributor",
    "rating": "Rating",
    "plot": "Plot",
    "story": "Plot",
    "about": "Plot",
    "goal": "Goal",
    "mission": "Goal",
    "time travel": "Time Travel",
    "iron man": "Iron Man",
    "tony": "Iron Man",
    "stark": "Iron Man",
    "captain america": "Captain America",
    "steve": "Captain America",
    "rogers": "Captain America",
    "thor": "Thor",
    "hulk": "Hulk",
    "banner": "Hulk",
    "black widow": "Black Widow",
    "natasha": "Black Widow",
    "hawkeye": "Hawkeye",
    "clint": "Hawkeye",
    "captain marvel": "Captain Marvel",
    "carol": "Captain Marvel",
    "ant man": "Ant Man",
    "scott": "Ant Man",
    "ending": "Ending",
    "end": "Ending",
    "how does it end": "Ending"
}

def retrieve_fact(query, df, translations):
    """Retrieves matching fact based on exact keywords or TF-IDF Cosine Similarity."""
    if df.empty:
        return None, 0.0
        
    q = query.lower().strip()
    
    # 1. Exact / Substring Keyword Matching
    for keyword, topic in KEYWORD_MAPPINGS.items():
        if keyword in q:
            # Verify if this topic exists in dataframe
            row = df[df['topic'].str.lower() == topic.lower()]
            if not row.empty:
                return row.iloc[0], 1.0
                
    # 2. Vectorized Matching using TF-IDF and Cosine Similarity
    # Build list of phrases and map them to canonical topics
    all_phrases = []
    phrase_to_topic = {}
    for topic, phrases in translations.items():
        for phrase in phrases:
            all_phrases.append(phrase)
            phrase_to_topic[phrase] = topic
            
    if not all_phrases:
        # If no translations, match directly to CSV topics
        for topic in df['topic'].unique():
            all_phrases.append(topic)
            phrase_to_topic[topic] = topic

    # Add user query to vocabulary space
    vectorizer = TfidfVectorizer().fit(all_phrases + [query])
    phrase_vectors = vectorizer.transform(all_phrases)
    query_vector = vectorizer.transform([query])
    
    similarities = cosine_similarity(query_vector, phrase_vectors).flatten()
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    
    # Threshold for validation
    if best_score > 0.12:
        best_phrase = all_phrases[best_idx]
        matched_topic = phrase_to_topic[best_phrase]
        row = df[df['topic'].str.lower() == matched_topic.lower()]
        if not row.empty:
            return row.iloc[0], float(best_score)
            
    return None, 0.0

# Sidebar Settings
st.sidebar.markdown(f"<h2 style='color:#ffc107;'>⚙️ Configurations</h2>", unsafe_allow_html=True)

# API Key configuration
api_key = st.sidebar.text_input("Google Gemini API Key", type="password", help="Enter your Gemini API key to enable natural RAG responses. If empty, the app will run in offline retrieval mode.")
model_name = st.sidebar.selectbox("Gemini Model", ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-1.5-pro"])

# About section
st.sidebar.markdown("---")
st.sidebar.markdown(f"<h3 style='color:#00ffcc;'>🛡️ About Avengers RAG</h3>", unsafe_allow_html=True)
st.sidebar.markdown("""
This app is a RAG (Retrieval-Augmented Generation) query system.
1. It parses questions.
2. It performs semantic search using **TF-IDF & Cosine Similarity** against mapped alternative phrasings.
3. It retrieves exact movie facts from the verified database.
4. It synthesizes natural answers using **Google Gemini** (if API key is supplied).
""")

# Show data statistics
st.sidebar.markdown("---")
if not df.empty:
    st.sidebar.markdown(f"**Database facts:** `{len(df)} topics`")
    st.sidebar.markdown(f"**Reference phrasings:** `{sum(len(v) for v in translations.values())} phrases`")

# Hero Header with background poster and overlay
st.markdown("""
<div class="hero-banner-container">
    <div class="hero-banner-bg"></div>
    <div class="hero-banner-overlay"></div>
    <div class="hero-banner-content">
        <h1 class="hero-banner-title">Avengers: Endgame Review Analyzer</h1>
        <p class="hero-banner-subtitle">Analyze Netflix movie reviews using Large Language Models (LLMs), NLP, and Sentiment Analysis.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Main container layout
col_main, col_stats = st.columns([7, 3])

with col_main:
    # Query box
    st.markdown("### 🔍 Search movie facts")
    
    # Set default text helper or recommendation click state
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""

    # Text Input
    query_input = st.text_input("Ask a question about the cast, runtime, plot, ending, time travel, directors, etc.", value=st.session_state.search_query)

    # Simple Recommendations / Quick Buttons
    st.markdown("**Quick Topics:**")
    rec_cols = st.columns(6)
    recommendations = ["Who directed?", "Main Cast", "What is the Ending?", "Tell me about Iron Man", "How does Time Travel work?", "Who is the Villain?"]
    
    for idx, rec in enumerate(recommendations):
        col_idx = idx % 6
        if rec_cols[col_idx].button(rec, key=f"rec_btn_{idx}", use_container_width=True):
            st.session_state.search_query = rec
            st.rerun()

    # Search Execution
    if query_input:
        with st.spinner("Retrieving facts..."):
            matched_row, score = retrieve_fact(query_input, df, translations)
            
        if matched_row is not None:
            topic = matched_row['topic']
            content = matched_row['content']
            
            st.markdown("### 🎯 Retrieval Result")
            
            # Display source block
            st.markdown(f"""
            <div class="fact-card">
                <div class="fact-card-title">
                    <span>📂 Verified Context Match: <b>{topic}</b></span>
                    <span class="metric-badge">Match Score: {score:.2f}</span>
                </div>
                <div class="fact-card-content">{content}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # RAG Generation Mode (If API key provided)
            if api_key:
                with st.spinner("Generating answer with Google Gemini..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel(model_name)
                        
                        prompt = f"""You are an expert assistant specialized in the movie "Avengers: Endgame".
Answer the user's question accurately and creatively using the verified context below.
Format your answer elegantly and keep it conversational.

Context:
Topic: {topic}
Verified Fact: {content}

User Question: {query_input}

Answer:"""
                        response = model.generate_content(prompt)
                        ai_response = response.text
                        
                        st.markdown("### 🔮 AI Generated Answer")
                        st.markdown(f"""
                        <div class="rag-card">
                            <div class="rag-card-title">🤖 Gemini Model Answer ({model_name})</div>
                            <div class="rag-card-content">{ai_response}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error calling Google Gemini API: {str(e)}")
                        st.info("Check your API key or model selection in the sidebar.")
            else:
                st.info("💡 **Tip:** Add a Google Gemini API Key in the sidebar to enable natural language generation / RAG answers!")
        else:
            st.warning("⚠️ No high-confidence facts matched your query. Try rephrasing or clicking one of the Quick Topics below.")
            
            # Suggest close topics
            st.markdown("#### Suggested Topics to Ask:")
            suggest_cols = st.columns(4)
            if not df.empty:
                random_topics = df['topic'].sample(min(8, len(df))).tolist()
                for idx, t_topic in enumerate(random_topics):
                    col_idx = idx % 4
                    if suggest_cols[col_idx].button(t_topic, key=f"sug_btn_{idx}", use_container_width=True):
                        st.session_state.search_query = f"Tell me about {t_topic}"
                        st.rerun()

with col_stats:
    st.markdown("### 📊 Database Explorer")
    if not df.empty:
        # Allow exploring the raw records
        selected_topic = st.selectbox("Select a Topic to browse", ["-- Select Topic --"] + list(df['topic'].unique()))
        
        if selected_topic != "-- Select Topic --":
            selected_row = df[df['topic'] == selected_topic].iloc[0]
            st.markdown(f"""
            <div class="fact-card" style="border-color: #ffc107;">
                <div class="fact-card-title" style="color: #ffc107;">📝 {selected_row['topic']}</div>
                <div class="fact-card-content">{selected_row['content']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("#### Database Preview")
        # Visual metrics
        cast_count = 0
        cast_row = df[df['topic'] == 'Main Cast']
        if not cast_row.empty:
            cast_count = len(str(cast_row.iloc[0]['content']).split(';'))
            
        m1, m2 = st.columns(2)
        m1.metric("Total Topics", len(df))
        m2.metric("Cast Members Listed", cast_count if cast_count > 0 else "N/A")
        
        st.dataframe(
            df[['topic', 'content']],
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No data in the database yet.")
