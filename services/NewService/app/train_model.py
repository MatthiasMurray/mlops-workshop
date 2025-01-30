import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def train_model():
    # Load the dataset
    df = pd.read_csv('app/data/house_prices.csv')
    
    # Basic preprocessing
    # Remove non-numeric columns and id
    exclude_cols = ['id', 'date']
    numeric_features = [col for col in df.columns if col not in exclude_cols and col != 'price']
    
    X = df[numeric_features]
    y = df['price']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    os.makedirs('app/model', exist_ok=True)
    with open('app/model/house_price_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'features': numeric_features
        }, f)
    
    # Print model performance
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"Train R² score: {train_score:.4f}")
    print(f"Test R² score: {test_score:.4f}")
    
    # Print feature importance
    feature_importance = dict(zip(numeric_features, model.feature_importances_))
    print("\nTop 5 most important features:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{feature}: {importance:.4f}")

if __name__ == "__main__":
    train_model() 