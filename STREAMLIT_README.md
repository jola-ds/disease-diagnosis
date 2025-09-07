# Disease Diagnosis Streamlit App

A comprehensive web application for showcasing the Disease Diagnosis AI project with interactive features for prediction, data analysis, and model performance visualization.

## Features

### 🏠 Home Page

- Project overview and key metrics
- Quick start guide
- Important disclaimers

### 🔮 Disease Prediction

- Interactive form for patient demographics
- Symptom selection interface
- Real-time disease prediction with confidence scores
- Probability distribution visualization

### 📊 Data Analysis

- Dataset overview and statistics
- Disease distribution charts
- Demographics analysis
- Symptom prevalence heatmap

### 📈 Model Performance

- Performance metrics and statistics
- Per-class F1-scores visualization
- Model architecture details

### ℹ️ About

- Project methodology and mission
- Target diseases and their performance
- Technical stack and disclaimers

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
# Option 1: Using the run script
python run_app.py

# Option 2: Direct streamlit command
streamlit run streamlit_app.py
```

### 3. Access the App

Open your browser and navigate to:

- Local: `http://localhost:8501`
- Network: `http://0.0.0.0:8501`

## Prerequisites

- Python 3.7+
- Trained model file (`models/final_model.pkl`)
- Synthetic dataset (`data/synthetic_dataset.csv`)

## Model Training

If you haven't trained the model yet, run:

```bash
python main.py
```

This will:

1. Load the synthetic dataset
2. Train the XGBoost model
3. Save the model to `models/final_model.pkl`
4. Generate evaluation metrics

## Data Generation

If you need to generate new synthetic data, you can use the data generator:

```python
from src.data_generator import NigerianDiseaseGenerator

generator = NigerianDiseaseGenerator()
df = generator.generate_dataset(n_patients=10000)
df.to_csv("data/synthetic_dataset.csv", index=False)
```

## App Structure

```
streamlit_app.py          # Main Streamlit application
run_app.py               # Script to run the app
STREAMLIT_README.md      # This file
requirements.txt         # Python dependencies
```

## Key Components

### Prediction Interface

- **Demographics**: Age group, gender, setting, region, season
- **Symptoms**: 15 binary symptom indicators
- **Output**: Predicted disease with confidence score and probability distribution

### Visualizations

- **Plotly Charts**: Interactive bar charts, pie charts, heatmaps
- **Real-time Updates**: Dynamic charts based on user input
- **Responsive Design**: Works on desktop and mobile devices

### Data Integration

- **Cached Loading**: Efficient data and model loading
- **Error Handling**: Graceful handling of missing files
- **Performance**: Optimized for fast loading and interaction

## Customization

### Adding New Diseases

1. Update the `class_names` list in `load_model()`
2. Modify the data generator to include new disease configurations
3. Retrain the model with the updated dataset

### Styling

- Modify the CSS in the `st.markdown()` sections
- Update color schemes in Plotly charts
- Customize the page layout and components

### Features

- Add new visualization types
- Implement additional prediction features
- Create new analysis pages

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure `models/final_model.pkl` exists
2. **Dataset not found**: Generate synthetic data or check file path
3. **Import errors**: Install all requirements with `pip install -r requirements.txt`
4. **Port conflicts**: Change the port in `run_app.py` or use `--server.port` flag

### Performance Tips

- Use `@st.cache_data` for expensive operations
- Limit data loading to necessary columns
- Optimize chart rendering for large datasets

## Security Notes

- This app is for educational purposes only
- No real patient data is used
- All data is synthetic and generated locally
- No external API calls or data transmission

## License

This project is part of the Disease Diagnosis AI project. See the main README.md for license information.
