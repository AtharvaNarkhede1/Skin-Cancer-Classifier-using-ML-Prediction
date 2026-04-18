"""
app.py
------
FastAPI backend for the skin cancer classification model.
Provides a /predict endpoint that accepts an image file.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
import tensorflow as tf

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_single_image

app = FastAPI(title="Skin Cancer Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model globably
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'skin_cancer_model.keras')
MODEL = None

@app.on_event("startup")
def load_model():
    global MODEL
    if os.path.exists(MODEL_PATH):
        try:
            MODEL = tf.keras.models.load_model(MODEL_PATH)
            print(f"✓ Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
    else:
        print(f"⚠ Warning: Model file not found at {MODEL_PATH}. Training might be required.")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "model_loaded": MODEL is not None,
        "api_name": "Skin Cancer Detection API"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        # Read file bytes
        contents = await file.read()
        
        # Preprocess
        input_arr = preprocess_single_image(contents)
        
        # Predict
        prediction_prob = MODEL.predict(input_arr)[0][0]
        
        # Binary classification threshold
        is_malignant = prediction_prob > 0.5
        label = "Malignant" if is_malignant else "Benign"
        confidence = float(prediction_prob if is_malignant else (1.0 - prediction_prob))
        
        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "raw_score": float(prediction_prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
