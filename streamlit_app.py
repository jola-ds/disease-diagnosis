import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from src.demo import predict_single_sample
from src.data_generator import NigerianDiseaseGenerator
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Disease Diagnosis AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with dark mode support
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-result {
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Dark mode specific overrides */
    @media (prefers-color-scheme: dark) {
        .main-header {
            color: #4a9eff;
        }
        .sub-header {
            color: #e0e0e0;
        }
        .metric-card {
            background-color: #2d2d2d;
            color: #e0e0e0;
            border-left-color: #4a9eff;
        }
        .prediction-result {
            background-color: #2d2d2d;
            color: #e0e0e0;
            border-color: #28a745;
        }
        .warning-box {
            background-color: #2d2d2d;
            color: #e0e0e0;
            border-left-color: #ffc107;
        }
        .info-box {
            background-color: #2d2d2d;
            color: #e0e0e0;
            border-left-color: #17a2b8;
        }
    }
    
    /* Streamlit dark theme specific overrides */
    .stApp[data-theme="dark"] .metric-card {
        background-color: #262730;
        color: #fafafa;
        border-left-color: #4a9eff;
    }
    .stApp[data-theme="dark"] .prediction-result {
        background-color: #262730;
        color: #fafafa;
        border-color: #28a745;
    }
    .stApp[data-theme="dark"] .warning-box {
        background-color: #262730;
        color: #fafafa;
        border-left-color: #ffc107;
    }
    .stApp[data-theme="dark"] .info-box {
        background-color: #262730;
        color: #fafafa;
        border-left-color: #17a2b8;
    }
    .stApp[data-theme="dark"] .main-header {
        color: #4a9eff;
    }
    .stApp[data-theme="dark"] .sub-header {
        color: #fafafa;
    }
    
    /* Prominent button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4, #4a9eff);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(31, 119, 180, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #4a9eff, #1f77b4);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(31, 119, 180, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 10px rgba(31, 119, 180, 0.3);
    }
    
    /* Dark theme button styling */
    .stApp[data-theme="dark"] .stButton > button {
        background: linear-gradient(90deg, #4a9eff, #6bb6ff);
        box-shadow: 0 4px 15px rgba(74, 158, 255, 0.3);
    }
    
    .stApp[data-theme="dark"] .stButton > button:hover {
        background: linear-gradient(90deg, #6bb6ff, #4a9eff);
        box-shadow: 0 6px 20px rgba(74, 158, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load model and class names
@st.cache_data
def load_model():
    """Load the trained model and return class names"""
    model_path = "models/final_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please run the training pipeline first.")
        return None, None
    
    class_names = [
        "diabetes", "gastroenteritis", "healthy", "hiv", "hypertension",
        "malaria", "measles", "peptic_ulcer", "pneumonia", "tuberculosis", "typhoid"
    ]
    return model_path, class_names

# Load data for visualization
@st.cache_data
def load_data():
    """Load the synthetic dataset for visualization"""
    data_path = "data/synthetic_dataset.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Disease Diagnosis AI</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Disease Prediction for Nigerian Healthcare")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔮 Disease Prediction", "📊 Data Analysis", "📈 Model Performance", "ℹ️ About"]
    )
    
    # Check for session state page (set by button click)
    if 'page' in st.session_state and st.session_state.page:
        page = st.session_state.page
        # Reset session state after using it
        st.session_state.page = None
    
    # Load model and data
    model_path, class_names = load_model()
    df = load_data()
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Disease Prediction":
        show_prediction_page(model_path, class_names)
    elif page == "📊 Data Analysis":
        show_data_analysis_page(df)
    elif page == "📈 Model Performance":
        show_model_performance_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    """Display the home page with project overview"""
    
    # Prominent prediction button at the top
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔮 Start Disease Prediction", type="primary", use_container_width=True, help="Click to go directly to the disease prediction tool"):
            st.session_state.page = "🔮 Disease Prediction"
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 Project Overview
    
    This AI-powered disease diagnosis system is designed to support healthcare workers in low-resource settings, 
    particularly in Nigeria. Using machine learning and synthetic data that reflects real epidemiological patterns, 
    the system can predict 10 common diseases based on patient demographics and symptoms.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Key features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>🎯 High Accuracy</h4>
        <p>74.6% overall accuracy with XGBoost classifier</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>🏥 10 Diseases</h4>
        <p>Malaria, Typhoid, TB, Measles, and more</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>📊 Real-time Predictions</h4>
        <p>Instant diagnosis with confidence scores</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Warning
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown("""
    ⚠️ **Important Disclaimer**: This system is for educational and research purposes only. 
    It should not be used as a substitute for professional medical diagnosis or treatment. 
    Always consult with qualified healthcare professionals for medical decisions.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick start
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    1. **Click the button at the top** or navigate to Disease Prediction using the sidebar
    2. **Enter patient demographics** (age, gender, location, season)
    3. **Select symptoms** the patient is experiencing
    4. **Get instant predictions** with confidence scores
    5. **Explore data analysis** to understand disease patterns
    """)

def show_prediction_page(model_path, class_names):
    """Display the disease prediction interface"""
    st.markdown('<h2 class="sub-header">🔮 Disease Prediction</h2>', unsafe_allow_html=True)
    
    if model_path is None:
        st.error("Model not available. Please ensure the model is trained first.")
        return
    
    # Create two columns for input form
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Patient Demographics")
        
        # Demographics inputs
        age_band = st.selectbox(
            "Age Group",
            ["0-4", "5-14", "15-24", "25-44", "45-64", "65+"],
            help="Select the patient's age group"
        )
        
        gender = st.selectbox(
            "Gender",
            ["male", "female"],
            help="Select the patient's gender"
        )
        
        setting = st.selectbox(
            "Setting",
            ["urban", "rural"],
            help="Urban or rural setting"
        )
        
        region = st.selectbox(
            "Region",
            ["north", "middle_belt", "south"],
            help="Geographic region in Nigeria"
        )
        
        season = st.selectbox(
            "Season",
            ["dry", "rainy", "transition"],
            help="Current season"
        )
    
    with col2:
        st.markdown("### Symptoms")
        st.markdown("Select all symptoms the patient is experiencing:")
        
        # Symptoms inputs
        symptoms = [
            "fever", "headache", "cough", "fatigue", "body_ache", "chills", "sweats",
            "nausea", "vomiting", "diarrhea", "abdominal_pain", "loss_of_appetite",
            "sore_throat", "runny_nose", "dysuria"
        ]
        
        symptom_values = {}
        for i, symptom in enumerate(symptoms):
            if i % 2 == 0:  # Create two columns for symptoms
                col_a, col_b = st.columns(2)
                with col_a:
                    symptom_values[symptom] = st.checkbox(symptom.replace("_", " ").title(), key=f"symptom_{symptom}")
            else:
                with col_b:
                    symptom_values[symptom] = st.checkbox(symptom.replace("_", " ").title(), key=f"symptom_{symptom}")
    
    # Prediction button
    if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
        # Prepare input data
        input_data = {
            "age_band": age_band,
            "gender": gender,
            "setting": setting,
            "region": region,
            "season": season,
            **{symptom: 1 if symptom_values[symptom] else 0 for symptom in symptoms}
        }
        
        # Make prediction
        try:
            result = predict_single_sample(input_data, model_path, class_names)
            
            # Display results
            st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Prediction Result")
            st.markdown(f"**Predicted Disease:** {result['predicted_disease'].replace('_', ' ').title()}")
            st.markdown(f"**Confidence:** {result['confidence']:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show all probabilities
            st.markdown("### 📊 All Disease Probabilities")
            prob_df = pd.DataFrame([
                {"Disease": disease.replace("_", " ").title(), "Probability": f"{prob:.1%}"}
                for disease, prob in result['all_probabilities'].items()
            ]).sort_values("Probability", ascending=False)
            
            # Create a bar chart
            fig = px.bar(
                prob_df, 
                x="Probability", 
                y="Disease", 
                orientation="h",
                title="Disease Probability Distribution",
                color="Probability",
                color_continuous_scale="Blues"
            )
            fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def show_data_analysis_page(df):
    """Display data analysis and visualization"""
    st.markdown('<h2 class="sub-header">📊 Data Analysis</h2>', unsafe_allow_html=True)
    
    if df is None:
        st.error("Dataset not available. Please ensure the synthetic dataset is generated.")
        return
    
    # Dataset overview
    st.markdown("### 📈 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", f"{len(df):,}")
    with col2:
        st.metric("Diseases", len(df['diagnosis'].unique()))
    with col3:
        st.metric("Symptoms", 15)
    with col4:
        st.metric("Features", len(df.columns))
    
    # Disease distribution
    st.markdown("### 🦠 Disease Distribution")
    
    disease_counts = df['diagnosis'].value_counts()
    disease_pct = df['diagnosis'].value_counts(normalize=True) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig_pie = px.pie(
            values=disease_counts.values,
            names=disease_counts.index,
            title="Disease Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart
        fig_bar = px.bar(
            x=disease_counts.index,
            y=disease_counts.values,
            title="Disease Counts",
            labels={'x': 'Disease', 'y': 'Count'},
            color=disease_counts.values,
            color_continuous_scale="Blues"
        )
        fig_bar.update_xaxis(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Demographics analysis
    st.markdown("### 👥 Demographics Analysis")
    
    demo_cols = ['age_band', 'gender', 'setting', 'region', 'season']
    fig_demo = make_subplots(
        rows=2, cols=3,
        subplot_titles=demo_cols,
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    for i, col in enumerate(demo_cols):
        row = i // 3 + 1
        col_pos = i % 3 + 1
        
        counts = df[col].value_counts()
        fig_demo.add_trace(
            go.Bar(x=counts.index, y=counts.values, name=col),
            row=row, col=col_pos
        )
    
    fig_demo.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_demo, use_container_width=True)
    
    # Symptom prevalence by disease
    st.markdown("### 🔍 Symptom Prevalence by Disease")
    
    symptoms = ['fever', 'headache', 'cough', 'fatigue', 'body_ache', 'chills', 'sweats',
                'nausea', 'vomiting', 'diarrhea', 'abdominal_pain', 'loss_of_appetite',
                'sore_throat', 'runny_nose', 'dysuria']
    
    # Create heatmap data
    heatmap_data = []
    for disease in df['diagnosis'].unique():
        disease_df = df[df['diagnosis'] == disease]
        row = []
        for symptom in symptoms:
            prevalence = disease_df[symptom].mean() * 100
            row.append(prevalence)
        heatmap_data.append(row)
    
    fig_heatmap = px.imshow(
        heatmap_data,
        x=symptoms,
        y=df['diagnosis'].unique(),
        color_continuous_scale="Blues",
        title="Symptom Prevalence by Disease (%)",
        labels=dict(x="Symptoms", y="Diseases", color="Prevalence %")
    )
    fig_heatmap.update_xaxis(tickangle=45)
    st.plotly_chart(fig_heatmap, use_container_width=True)

def show_model_performance_page():
    """Display model performance metrics"""
    st.markdown('<h2 class="sub-header">📈 Model Performance</h2>', unsafe_allow_html=True)
    
    # Model metrics
    st.markdown("### 🎯 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>Overall Accuracy</h4>
        <h2>74.6%</h2>
        <p>On held-out test set</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>Macro F1-Score</h4>
        <h2>0.70</h2>
        <p>Average across all classes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>Weighted F1-Score</h4>
        <h2>0.75</h2>
        <p>Balanced performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Per-class performance
    st.markdown("### 🏆 Per-Class Performance")
    
    performance_data = {
        "Disease": ["Malaria", "Measles", "Healthy", "Tuberculosis", "Typhoid", 
                   "Gastroenteritis", "Pneumonia", "Peptic Ulcer", "Diabetes", 
                   "Hypertension", "HIV"],
        "F1-Score": [0.86, 0.82, 0.89, 0.66, 0.68, 0.75, 0.78, 0.72, 0.65, 0.47, 0.70],
        "Category": ["Strong", "Strong", "Strong", "Moderate", "Moderate", 
                    "Good", "Good", "Good", "Moderate", "Lower", "Good"]
    }
    
    perf_df = pd.DataFrame(performance_data)
    
    # Color mapping for categories
    color_map = {"Strong": "#28a745", "Good": "#17a2b8", "Moderate": "#ffc107", "Lower": "#dc3545"}
    perf_df['Color'] = perf_df['Category'].map(color_map)
    
    fig_perf = px.bar(
        perf_df,
        x="Disease",
        y="F1-Score",
        color="Category",
        title="F1-Score by Disease",
        color_discrete_map=color_map
    )
    fig_perf.update_layout(height=500, xaxis_tickangle=45)
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # Model details
    st.markdown("### 🔧 Model Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Algorithm:** XGBoost Classifier
        
        **Features:**
        - Demographics (age, gender, setting, region, season)
        - 15 binary symptom indicators
        - One-hot encoded categorical variables
        
        **Training:**
        - 80% training, 20% testing split
        - Stratified sampling
        - Sample weights for class imbalance
        """)
    
    with col2:
        st.markdown("""
        **Hyperparameters:**
        - n_estimators: 200
        - max_depth: 6
        - learning_rate: 0.1
        - random_state: 42
        
        **Validation:**
        - Bootstrap confidence intervals
        - Cross-validation metrics
        - Realistic data patterns
        """)

def show_about_page():
    """Display about page with project information"""
    st.markdown('<h2 class="sub-header">ℹ️ About This Project</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Mission Statement
    
    This project addresses the urgent need for accessible, privacy-safe, and context-aware disease prediction 
    tools in low-resource healthcare settings, particularly in Nigeria. Using domain-informed synthetic data 
    and machine learning, we demonstrate how AI can support healthcare decision-making.
    """)
    
    st.markdown("""
    ### 🔬 Methodology
    
    **Data Generation:**
    - Synthetic Nigerian hospital data with realistic epidemiological patterns
    - 10 diseases + healthy cases
    - 15 common symptoms with disease-specific probabilities
    - Demographic adjustments based on real population distributions
    - Seasonal effects and symptom correlations
    
    **Machine Learning:**
    - XGBoost classifier with 74.6% accuracy
    - One-hot encoding for categorical variables
    - Sample weights to handle class imbalance
    - Bootstrap confidence intervals for uncertainty estimation
    """)
    
    st.markdown("""
    ### 🏥 Target Diseases
    
    The system can predict the following diseases:
    
    - **Malaria** (F1: 0.86) - Vector-borne, seasonal
    - **Typhoid** (F1: 0.68) - Water-borne, rural predominance  
    - **Tuberculosis** (F1: 0.66) - Airborne, dry season peak
    - **Measles** (F1: 0.82) - Pediatric, highly contagious
    - **Gastroenteritis** (F1: 0.75) - Food/water-borne
    - **Pneumonia** (F1: 0.78) - Respiratory, seasonal
    - **Peptic Ulcer** (F1: 0.72) - Adult-onset, urban
    - **Diabetes** (F1: 0.65) - Chronic, lifestyle-related
    - **Hypertension** (F1: 0.47) - Chronic, age-related
    - **HIV** (F1: 0.70) - Infectious, demographic patterns
    - **Healthy** (F1: 0.89) - No disease detected
    """)
    
    st.markdown("""
    ### ⚠️ Important Disclaimers
    
    - **Educational Purpose Only**: This system is for research and educational purposes
    - **Not Medical Advice**: Never substitute for professional medical diagnosis
    - **Synthetic Data**: Based on simulated data, not real patient records
    - **Research Tool**: Intended to demonstrate AI potential in healthcare
    - **Privacy Safe**: No real patient data is used or stored
    """)
    
    st.markdown("""
    ### 📚 References
    
    This project is based on research into:
    - Nigerian disease epidemiology and patterns
    - Machine learning in healthcare applications
    - Disease prediction in low-resource settings
    - Synthetic data generation for medical research
    
    For full citations, see the `citations.md` file in the project repository.
    """)
    
    st.markdown("""
    ### 🛠️ Technical Stack
    
    - **Python 3.x** - Core programming language
    - **XGBoost** - Machine learning algorithm
    - **scikit-learn** - Data preprocessing and evaluation
    - **Streamlit** - Web application framework
    - **Plotly** - Interactive visualizations
    - **Pandas/NumPy** - Data manipulation
    """)

if __name__ == "__main__":
    main()
