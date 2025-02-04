#!/bin/bash

echo "Starting model training..."
python app/train_model.py

echo "Training complete. Model is available at app/model/house_price_model.pkl"