import io
import os
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
import torchvision 
import boto3
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

S3_BUCKET_NAME = "your-bucket-name"
S3_MODEL_KEY = "path/to/your/model.pkl"  # e.g., "models/random_forest.pkl"
LOCAL_MODEL_PATH = "/tmp/current_model.pkl"  # Store temporarily


DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None

def download_model():
    s3 = boto3.client('s3')
    s3.download_file(S3_BUCKET_NAME, S3_MODEL_KEY, LOCAL_MODEL_PATH)
    print("Download complete. Loading model...")


def load_model_from_s3():
    """Downloads model from S3 and loads it into the global variable."""
    global model
    try:
        download_model
        # Load the model (adjust based on your model's serialization format)
        model = torch.load(LOCAL_MODEL_PATH,map_location = DEVICE)
        model.eval()

        print("Model loaded successfully and ready for inference.")

    except Exception as e:
        print(f"FATAL: Could not load model from S3: {e}")
        # In production, you might want to raise an exception to prevent the app from starting
        model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    """
    # Startup: Load the model
    load_model_from_s3()
    yield
    # Shutdown: Clean up (if necessary)
    if os.path.exists(LOCAL_MODEL_PATH):
        os.remove(LOCAL_MODEL_PATH)
        print("Cleaned up temporary model file.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Preprocessing ─────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    # Check model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Read and validate image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    # Preprocess
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # Inference
    try:
        with torch.no_grad():
            output = model(tensor)
        prediction = torch.argmax(output, dim=1).item()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {"prediction": prediction}