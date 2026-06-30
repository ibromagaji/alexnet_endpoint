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
import json
import httpx 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

S3_BUCKET_NAME = "vgg_model"
S3_MODEL_KEY = "vgg_model.ckpt"  # e.g., "models/random_forest.pkl"
LOCAL_MODEL_PATH = "/tmp/vgg_model.ckpt" # Store temporarily
S3_BUCKET_CLASSNAMES_NAME = 'alexnet-class-names'
S3_CLASSNAME_FILE = '/class_names'
LOCAL_CLASSNAME_FILE = '/tmp/class_names'


DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
idx_class = None

def download_model():
    s3 = boto3.client('s3')
    s3.download_file(S3_BUCKET_NAME, S3_MODEL_KEY, LOCAL_MODEL_PATH)
    print("Download complete. Loading model...")

def download_classnames():
    s3 = boto3.client('s3')
    s3.download_file(S3_BUCKET_CLASSNAMES_NAME,S3_CLASSNAME_FILE,LOCAL_CLASSNAME_FILE)
    
                      


def load_model_from_s3():
    """Downloads model from S3 and loads it into the global variable."""
    global model,idx_class
    try:
        # Use the authenticated s3_client, not a new bare client
        print(f"Downloading model from s3://{S3_BUCKET_NAME}/{S3_MODEL_KEY} ...")
        s3_client.download_file(S3_BUCKET_NAME, S3_MODEL_KEY, LOCAL_MODEL_PATH)
        print("Download complete. Loading model...")

        model = models.alexnet(weights=None,num_classes = 29)

        state_dict = torch.load(LOCAL_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully and ready for inference.")

    except Exception as e:
        print(f"FATAL: Could not load model from S3: {e}")
        model = None

    try:
        s3_client.download_file(S3_BUCKET_CLASSNAMES_NAME,S3_CLASSNAME_FILE,LOCAL_CLASSNAME_FILE)
        # Load class names from downloaded JSON
        with open(LOCAL_CLASSNAME_FILE, "r") as f:
            idx_class = json.load(f)
        print('Downloaded class names successfully')

    except Exception as e:
        print(f'ERROR:Could not download class names: {e}')
        idx_class = None
        


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

    disease_name = idx_class[str[str(prediction)]]

    try:

        # Replace with your actual Gemini API key
        GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

        # The standard REST endpoint format for Gemini
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={GEMINI_API_KEY}"

        async with httpx.AsyncClient() as client:
            llm_resp = await client.post(
                url,
                headers={
                    "Content-Type": "application/json"
                },
                json={
                "systemInstruction": {
                    "parts": [
                        {"text":f"""You are a plant pathology expert. The disease identified is: "{disease_name}"

                            Return ONLY a valid JSON object with exactly these keys, nothing else:
                            {{
                            "description": "<2-3 sentence overview>",
                            "symptoms":    ["<s1>", "<s2>", "<s3>", "<s4>"],
                            "treatment":   ["<t1>", "<t2>", "<t3>", "<t4>"],
                            "prevention":  ["<p1>", "<p2>", "<p3>", "<p4>"]
                            }}
                            No markdown, no backticks. Raw JSON only."""
                            }
                    ]
                    },
                "contents": [
                    {
                    "parts": [
                        {"text": disease_name}
                    ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": 800,
                    "responseMimeType": "application/json" # Forces clean JSON output
                    }
                },
                timeout=15.0
            )

        # Parsing the response directly
        resp_json = llm_resp.json()
        raw = resp_json["candidates"][0]["content"]["parts"][0]["text"].strip()

        # No more backtick stripping needed! You can load it safely right away:
        info = json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    return {"prediction": disease_name,
            'disease_info':info
            }








    # return {
    # "prediction": prediction,
    # "class_name": idx_class[str(prediction)]  # JSON keys are strings!
    # }
