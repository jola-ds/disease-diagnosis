# Disease Diagnosis API

A FastAPI-based REST API for disease prediction using machine learning. This API serves a trained XGBoost model that predicts diseases based on patient symptoms and demographics.

## 🚀 Quick Start

### 1. Train the Model

```bash
python main.py
```

### 2. Start the API Server

```bash
python run_api.py
```

### 3. Test the API

```bash
python test_api.py
```

## 📖 API Documentation

Once the server is running, visit:

- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

## 🔗 Endpoints

### Health Check

- **GET** `/health` - Check API health and model status

### Information

- **GET** `/` - API information and available endpoints
- **GET** `/diseases` - List all possible diseases
- **GET** `/model/info` - Model information and performance metrics

### Predictions

- **POST** `/predict` - Predict disease for a single patient
- **POST** `/predict/batch` - Predict diseases for multiple patients

## 📝 Usage Examples

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age_band": "25-44",
       "gender": "female",
       "setting": "urban",
       "region": "north",
       "season": "dry",
       "fever": 1,
       "headache": 1,
       "cough": 0,
       "fatigue": 1,
       "body_ache": 1,
       "chills": 1,
       "sweats": 0,
       "nausea": 0,
       "vomiting": 0,
       "diarrhea": 0,
       "abdominal_pain": 0,
       "loss_of_appetite": 1,
       "sore_throat": 0,
       "runny_nose": 0,
       "dysuria": 0
     }'
```

**Response:**

```json
{
	"predicted_disease": "malaria",
	"confidence": 0.85,
	"all_probabilities": {
		"diabetes": 0.02,
		"gastroenteritis": 0.01,
		"healthy": 0.05,
		"hiv": 0.01,
		"hypertension": 0.01,
		"malaria": 0.85,
		"measles": 0.02,
		"peptic_ulcer": 0.01,
		"pneumonia": 0.01,
		"tuberculosis": 0.01,
		"typhoid": 0.01
	},
	"timestamp": "2024-01-15T10:30:00.000Z"
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "patients": [
         {
           "age_band": "25-44",
           "gender": "female",
           "setting": "urban",
           "region": "north",
           "season": "dry",
           "fever": 1,
           "headache": 1,
           "cough": 0,
           "fatigue": 1,
           "body_ache": 1,
           "chills": 1,
           "sweats": 0,
           "nausea": 0,
           "vomiting": 0,
           "diarrhea": 0,
           "abdominal_pain": 0,
           "loss_of_appetite": 1,
           "sore_throat": 0,
           "runny_nose": 0,
           "dysuria": 0
         },
         {
           "age_band": "5-14",
           "gender": "male",
           "setting": "rural",
           "region": "south",
           "season": "rainy",
           "fever": 0,
           "headache": 0,
           "cough": 0,
           "fatigue": 1,
           "body_ache": 0,
           "chills": 0,
           "sweats": 0,
           "nausea": 0,
           "vomiting": 0,
           "diarrhea": 0,
           "abdominal_pain": 0,
           "loss_of_appetite": 0,
           "sore_throat": 0,
           "runny_nose": 0,
           "dysuria": 0
         }
       ]
     }'
```

## 🏗️ Architecture

### Model

- **Algorithm**: XGBoost Classifier
- **Accuracy**: 74.6% on test set
- **Classes**: 11 diseases + healthy
- **Features**: 20 (5 demographics + 15 symptoms)

### API Features

- **FastAPI**: Modern, fast web framework
- **Pydantic**: Data validation and serialization
- **CORS**: Cross-origin resource sharing enabled
- **Auto-documentation**: Interactive API docs
- **Error handling**: Comprehensive error responses
- **Type hints**: Full type safety

## 🔧 Configuration

### Environment Variables

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `MODEL_PATH`: Path to model file (default: models/final_model.pkl)

### CORS Settings

The API is configured to allow all origins for development. For production, update the CORS settings in `api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Production domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 🧪 Testing

### Automated Tests

```bash
python test_api.py
```

### Manual Testing

1. Start the server: `python run_api.py`
2. Visit http://localhost:8000/docs
3. Use the interactive interface to test endpoints

### Test Coverage

- Health check endpoint
- Root endpoint
- Diseases list endpoint
- Model info endpoint
- Single prediction endpoint
- Batch prediction endpoint

## 📊 Model Performance

### Overall Metrics

- **Accuracy**: 74.6%
- **Macro F1-score**: 0.71
- **Weighted F1-score**: 0.75

### Per-Class Performance

- **Strong performance**: Malaria (F1=0.86), Measles (F1=0.82), Healthy (F1=0.89)
- **Moderate performance**: Tuberculosis (F1=0.66), Typhoid (F1=0.68)
- **Lower performance**: Hypertension (recall=0.47)

## 🚨 Important Notes

### Medical Disclaimer

⚠️ **This API is for research and educational purposes only. It should not be used for actual medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals.**

### Data Privacy

- The API processes patient data in memory only
- No data is stored or logged
- All predictions are stateless

### Production Considerations

- Add authentication and authorization
- Implement rate limiting
- Add logging and monitoring
- Use HTTPS in production
- Configure proper CORS settings
- Add input validation and sanitization
- Implement caching for better performance

## 🛠️ Development

### Project Structure

```
disease-diagnosis/
├── api.py                 # FastAPI application
├── run_api.py            # Server startup script
├── test_api.py           # API test suite
├── main.py               # Model training script
├── models/               # Trained model files
│   └── final_model.pkl
├── src/                  # Source code
│   ├── train.py          # Training utilities
│   ├── eval.py           # Evaluation utilities
│   └── demo.py           # Demo utilities
└── requirements.txt      # Dependencies
```

### Adding New Features

1. Update the Pydantic models for new input/output fields
2. Modify the prediction logic in the endpoints
3. Update the test suite
4. Update this documentation

## 📞 Support

For questions or issues:

1. Check the API documentation at `/docs`
2. Review the test suite for usage examples
3. Check the model training notebook for data understanding
4. Review the source code for implementation details
