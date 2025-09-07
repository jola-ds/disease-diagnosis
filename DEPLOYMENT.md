# Heroku Deployment Guide

This guide will help you deploy the Disease Diagnosis API to Heroku.

## 🚀 **Quick Deployment**

### Prerequisites

1. [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) installed
2. Git repository initialized
3. Heroku account

### Step 1: Prepare Your Repository

```bash
# Initialize git if not already done
git init

# Add all files
git add .

# Commit your changes
git commit -m "Initial commit with API and model"
```

### Step 2: Create Heroku App

```bash
# Login to Heroku
heroku login

# Create a new Heroku app
heroku create your-app-name

# Or create with a specific region
heroku create your-app-name --region us
```

### Step 3: Configure Environment Variables (Optional)

```bash
# Set allowed origins for production (optional)
heroku config:set ALLOWED_ORIGINS="https://yourdomain.com,https://www.yourdomain.com"

# Set any other environment variables
heroku config:set MODEL_PATH="models/final_model.pkl"
```

### Step 4: Deploy

```bash
# Deploy to Heroku
git push heroku main

# Or if using master branch
git push heroku master
```

### Step 5: Verify Deployment

```bash
# Check app status
heroku ps

# View logs
heroku logs --tail

# Open your app in browser
heroku open
```

## 📁 **Deployment Files**

### `Procfile`

```
web: uvicorn api:app --host 0.0.0.0 --port $PORT
```

### `runtime.txt`

```
python-3.11.10
```

### `requirements.txt`

```
# Core stack
numpy
pandas

# Machine learning
scikit-learn
xgboost
joblib

# Visualization
matplotlib
seaborn

# Notebook environment
jupyter

# App/demo
streamlit

# API
fastapi
uvicorn
pydantic
```

## 🔧 **Configuration**

### Environment Variables

- `ALLOWED_ORIGINS`: Comma-separated list of allowed CORS origins
- `MODEL_PATH`: Path to the model file (default: models/final_model.pkl)

### Heroku-Specific Settings

- **Port**: Heroku automatically sets the `$PORT` environment variable
- **Host**: Must be `0.0.0.0` to bind to all interfaces
- **Python Version**: Specified in `runtime.txt`

## 🚨 **Important Notes**

### Model File Size

- Heroku has a 500MB slug size limit
- The XGBoost model should be well within this limit
- If you encounter size issues, consider:
  - Using model compression
  - Storing the model in cloud storage (S3, etc.)
  - Loading the model from external source

### Memory Considerations

- Heroku free tier has 512MB RAM
- XGBoost models can be memory-intensive
- Consider upgrading to a paid tier for production

### Security

- Update CORS settings for production
- Consider adding authentication
- Use HTTPS in production

## 🧪 **Testing Your Deployment**

### Health Check

```bash
curl https://your-app-name.herokuapp.com/health
```

### API Documentation

Visit: `https://your-app-name.herokuapp.com/docs`

### Sample Prediction

```bash
curl -X POST "https://your-app-name.herokuapp.com/predict" \
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

## 🔄 **Updating Your App**

```bash
# Make changes to your code
# Commit changes
git add .
git commit -m "Update API"

# Deploy updates
git push heroku main
```

## 📊 **Monitoring**

### View Logs

```bash
# Real-time logs
heroku logs --tail

# Recent logs
heroku logs --num 100
```

### Check App Status

```bash
# App status
heroku ps

# App info
heroku info
```

### Scale Your App

```bash
# Scale to 1 web dyno
heroku ps:scale web=1

# Scale to 2 web dynos
heroku ps:scale web=2
```

## 🛠️ **Troubleshooting**

### Common Issues

1. **Build Fails**

   - Check `requirements.txt` for missing dependencies
   - Ensure Python version in `runtime.txt` is supported
   - Check build logs: `heroku logs --tail`

2. **App Crashes**

   - Check if model file exists
   - Verify all dependencies are installed
   - Check memory usage

3. **CORS Issues**

   - Update `ALLOWED_ORIGINS` environment variable
   - Check browser console for errors

4. **Model Loading Issues**
   - Ensure model file is committed to git
   - Check file path in code
   - Verify model file size

### Debug Commands

```bash
# Run one-off dyno for debugging
heroku run bash

# Check environment variables
heroku config

# Restart app
heroku restart
```

## 🌐 **Custom Domain (Optional)**

```bash
# Add custom domain
heroku domains:add www.yourdomain.com

# Configure DNS
# Point your domain to your Heroku app
```

## 💰 **Pricing Considerations**

- **Free Tier**: Limited hours, basic features
- **Hobby Tier**: $7/month, always on, custom domains
- **Standard Tier**: $25+/month, scaling, monitoring

## 🔒 **Production Checklist**

- [ ] Update CORS settings
- [ ] Set up monitoring
- [ ] Configure custom domain
- [ ] Set up SSL/HTTPS
- [ ] Add authentication if needed
- [ ] Set up logging
- [ ] Configure backup strategy
- [ ] Test all endpoints
- [ ] Load test the API
- [ ] Set up alerts

## 📞 **Support**

- [Heroku Documentation](https://devcenter.heroku.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Heroku Support](https://help.heroku.com/)
