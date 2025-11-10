# Deploy to Streamlit Cloud - Step by Step

Your repository is now ready for deployment!

## Step 1: Go to Streamlit Cloud

1. Open your browser and go to: https://share.streamlit.io/
2. Click the "Sign in" button in the top right
3. Sign in with your GitHub account (likthvishal)

## Step 2: Create New App

1. After signing in, click "New app" button
2. You'll see a form with three fields:
   - Repository: Select `likthvishal/glaucoma-detection`
   - Branch: Select `main`
   - Main file path: Enter `streamlit_app.py`

## Step 3: Deploy

1. Click the "Deploy!" button
2. Wait 5-10 minutes for the initial deployment
3. Streamlit will:
   - Install all packages from requirements.txt
   - Load your trained model from checkpoints/
   - Start the web application

## Step 4: Access Your App

Once deployed, your app will be available at:
```
https://glaucoma-detection-[your-app-id].streamlit.app
```

You can share this URL with anyone!

## What Happens During Deployment

Streamlit Cloud will:
- Clone your GitHub repository
- Download the model files via Git LFS
- Install TensorFlow, OpenCV, and other dependencies
- Start the Streamlit server
- Make your app publicly accessible

## App Features

Your deployed app includes:
- Upload retinal images (PNG, JPG, JPEG)
- Real-time glaucoma detection
- Confidence scores
- Grad-CAM visualization showing model focus areas
- Professional medical interface
- Mobile-responsive design

## Troubleshooting

If deployment fails:

1. Check the logs in Streamlit Cloud dashboard
2. Verify requirements.txt has all dependencies
3. Ensure Git LFS uploaded model files correctly

## Managing Your App

From Streamlit Cloud dashboard:
- View app logs
- Restart the app
- Update settings
- View analytics
- Delete the app

## Updating Your App

To update the deployed app:
```bash
git add .
git commit -m "Your update message"
git push origin main
```

Streamlit Cloud will automatically redeploy!

## Your Repository

https://github.com/likthvishal/glaucoma-detection

Everything is ready for deployment!
