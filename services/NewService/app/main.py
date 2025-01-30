import os
import logging
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
# import spacy

# # Load the trained spaCy model
# MODEL_PATH = "textcat_goemotions/training/cnn/model-best"  # Path to your trained model
# try:
#     nlp = spacy.load(MODEL_PATH)
# except Exception as e:
#     raise RuntimeError(f"Failed to load spaCy model: {e}")

service_name = os.getenv("SERVICE_NAME", "house_price_predictor")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(service_name)

# Load the model at startup
try:
    with open('app/model/house_price_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['features']
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError("Failed to load model")

class HouseFeatures(BaseModel):
    bedrooms: float = Field(..., description="Number of bedrooms")
    bathrooms: float = Field(..., description="Number of bathrooms")
    sqft_living: float = Field(..., description="Square footage of living space")
    sqft_lot: float = Field(..., description="Square footage of lot")
    floors: float = Field(..., description="Number of floors")
    waterfront: int = Field(..., description="Waterfront property (0 or 1)")
    condition: int = Field(..., description="Property condition (1-5)")
    grade: float = Field(..., description="Overall grade of the house (1-13)")
    sqft_above: float = Field(..., description="Square footage above ground")
    sqft_basement: Optional[float] = Field(0, description="Square footage of basement")
    yr_built: int = Field(..., description="Year the house was built")
    yr_renovated: Optional[int] = Field(0, description="Year of last renovation (0 if never renovated)")
    zipcode: int = Field(..., description="ZIP code")
    lat: float = Field(..., description="Latitude")
    long: float = Field(..., description="Longitude")
    sqft_living15: float = Field(..., description="Square footage of living space for nearest 15 neighbors")
    sqft_lot15: float = Field(..., description="Square footage of lot for nearest 15 neighbors")

class HealthResponse(BaseModel):
    status: str

class PredictionResponse(BaseModel):
    predicted_price: float
    top_influential_features: Dict[str, float]

class FeatureInfo(BaseModel):
    required_features: List[str]
    feature_ranges: Dict[str, Dict[str, str]]

logger.info(f"Service {service_name} started")

@app.get(f"/{service_name}/health", response_model=HealthResponse)
async def health_check():
    logger.info("Health check")
    return {"status": "ok"}

@app.post(f"/{service_name}/predict", response_model=PredictionResponse)
async def predict(house: HouseFeatures):
    try:
        # Convert house features to dictionary
        house_dict = house.dict()
        
        # Validate categorical values
        if house_dict['waterfront'] not in [0, 1]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value for waterfront. Valid values are: 0 (No) or 1 (Yes)"
            )
        if not 1 <= house_dict['condition'] <= 5:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value for condition. Valid values are: 1-5"
            )
        if not 1 <= house_dict['grade'] <= 13:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value for grade. Valid values are: 1-13"
            )
        
        # Create feature vector in the correct order
        feature_vector = []
        for feature in feature_names:
            value = house_dict.get(feature, 0)
            feature_vector.append(value)
        
        # Convert to numpy array and reshape
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Scale features
        scaled_features = scaler.transform(feature_vector)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        # Get feature importances for this prediction
        importances = dict(zip(feature_names, model.feature_importances_))
        top_features = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3])
        
        logger.info(f"Prediction made: ${prediction:,.2f}")
        return {
            "predicted_price": round(prediction, 2),
            "top_influential_features": top_features
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"/{service_name}/features", response_model=FeatureInfo)
async def get_features():
    """Return the list of features needed for prediction and their valid values"""
    feature_info = {
        "required_features": feature_names,
        "feature_ranges": {
            "waterfront": {"type": "categorical", "values": "0 (No) or 1 (Yes)"},
            "condition": {"type": "categorical", "values": "1-5 (Poor to Excellent)"},
            "grade": {"type": "categorical", "values": "1-13 (Poor to Luxury)"}
        }
    }
    return feature_info
