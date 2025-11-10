# Streamlit Deployment Guide

## Local Testing

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Ensure Model is Trained
Make sure you have trained the model first:
```bash
python train_glaucoma_model.py
```

This will create a `checkpoints/` folder with your trained model.

### Step 3: Run Streamlit App
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Deploying to Streamlit Cloud

### Prerequisites
- GitHub account with your repository
- Streamlit Cloud account (free at https://streamlit.io/cloud)

### Step 1: Prepare Your Repository
Your repository must include:
- [x] streamlit_app.py
- [x] requirements.txt
- [x] Trained model in checkpoints/ folder

**IMPORTANT**: The trained model files are currently in .gitignore. For deployment, you need to:

Option A: Remove model files from .gitignore temporarily
```bash
# Edit .gitignore and comment out these lines:
# *.keras
# checkpoints/
```

Option B: Use Git LFS for large files
```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes
git add checkpoints/
git commit -m "Add trained model for deployment"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Connect your GitHub account
4. Select your repository: `likthvishal/glaucoma-detection`
5. Set main file path: `streamlit_app.py`
6. Click "Deploy"

### Step 3: Wait for Deployment
- Initial deployment takes 5-10 minutes
- Streamlit Cloud will install all requirements
- Your app will be available at: `https://[your-app-name].streamlit.app`

## Alternative Deployment Options

### Option 1: Heroku
```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port=$PORT" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

### Option 2: Docker
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t glaucoma-detection .
docker run -p 8501:8501 glaucoma-detection
```

### Option 3: AWS/GCP/Azure
For cloud platforms, use Docker deployment or virtual machines with the local testing setup.

## Troubleshooting

### Model Not Found Error
If you see "Model not found!" in the app:
- Ensure checkpoints/ folder exists
- Ensure best_model.keras file is present
- Check file paths in streamlit_app.py (line 127-137)

### Out of Memory Error
If deployment fails due to memory:
- Streamlit Cloud free tier has 1GB RAM limit
- Consider using smaller model or paid tier
- Optimize model size with quantization

### Import Errors
If packages fail to import:
- Verify all dependencies are in requirements.txt
- Check Python version compatibility
- Streamlit Cloud uses Python 3.9-3.11

## Model Size Considerations

The trained model (checkpoints/) can be large (50-200MB). For GitHub:
- Free tier limit: 100MB per file
- Use Git LFS for files >50MB
- Consider model compression for deployment

## Security Notes

- This is a research/educational tool
- Add authentication if deploying publicly
- Do not use for actual medical diagnosis
- Include appropriate disclaimers

## App Features

The deployed app includes:
- Image upload (PNG, JPG, JPEG)
- Real-time prediction
- Confidence scores
- Grad-CAM visualization
- Interactive settings
- Mobile-responsive design

## Support

For issues:
1. Check model is properly trained
2. Verify all files are committed
3. Review Streamlit Cloud logs
4. Test locally first

---

Your app is now ready for deployment!
