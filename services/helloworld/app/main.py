import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
#import spacy

# Load the trained spaCy model
# MODEL_PATH = "textcat_goemotions/training/cnn/model-best"  # Path to your trained model
# try:
#     nlp = spacy.load(MODEL_PATH)
# except Exception as e:
    # raise RuntimeError(f"Failed to load spaCy model: {e}")

service_name = os.getenv("SERVICE_NAME", "default_service")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(service_name)

class InputText(BaseModel):
    text: str

logger.info(f"Service {service_name} started")

@app.get(f"/{service_name}/health")
async def health_check():
    logger.info("Health check")
    return {"status": "ok"}

@app.get(f"/{service_name}/predict")
async def predict():#input: InputText):
    try:
        #logger.info(f"Predicting for text: {input.text}")
        # doc = nlp(input.text)
        # return {"text": input.text, "labels": doc.cats}
        return {"answer": "helloworld"}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
