from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Dict, Any
import xgboost as xgb
import pandas as pd
import numpy as np
import shap
import uvicorn
from schemas import var_order, Input_schema_1, Input_schema_2, Input_schema_3, Output_schema
# 1. Define the Strict Orders (Replace these with your exact lists)
# This replaces the need for metadata.json
VARIABLE_ORDER_M1 = var_order[:-2]

# Model 2 adds G1, Model 3 adds G2
VARIABLE_ORDER_M2 = VARIABLE_ORDER_M1 + ["G1"]
VARIABLE_ORDER_M3 = VARIABLE_ORDER_M2 + ["G2"]

# Define which columns must be cast to category for XGBoost
CATEGORICAL_COLS = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 
                    'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'subject']

# Global dictionary to hold models in memory
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the models into memory at startup
    ml_models["Pre-G1"] = xgb.XGBRegressor()
    ml_models["Pre-G1"].load_model("models/model_1.json")
    
    ml_models["Pre-G2"] = xgb.XGBRegressor()
    ml_models["Pre-G2"].load_model("models/model_2.json")
    
    ml_models["Pre-Final"] = xgb.XGBRegressor()
    ml_models["Pre-Final"].load_model("models/model_3.json")
    
    # Initialize SHAP explainers to save time during requests
    ml_models["Explainer_M1"] = shap.TreeExplainer(ml_models["Pre-G1"])
    ml_models["Explainer_M2"] = shap.TreeExplainer(ml_models["Pre-G2"])
    ml_models["Explainer_M3"] = shap.TreeExplainer(ml_models["Pre-Final"])
    
    yield
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(lifespan=lifespan, title="Student Grade API")


@app.post('/predict_1', response_model=Output_schema)
def predict_1(request : Input_schema_1) :

    df = pd.DataFrame([request.model_dump()])
    df = df[VARIABLE_ORDER_M1] # This step aligns the data perfectly
        
    #  Apply category dtypes for XGBoost native support
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Generate Prediction
    pred = ml_models["Pre-G1"].predict(df)[0]
    pred = np.round(pred,2)

    # Generate SHAP values
    shap_val = ml_models["Explainer_M1"].shap_values(df)

    return Output_schema(my_prediction=pred, 
                         shap_values=shap_val[0].tolist(),
                         feature_names=VARIABLE_ORDER_M1)


@app.post('/predict_2', response_model=Output_schema)
def predict_2(request : Input_schema_2) :
    df = pd.DataFrame([request.model_dump()])
    df = df[VARIABLE_ORDER_M2] # This step aligns the data perfectly
        
    # Apply category dtypes for XGBoost native support
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Generate Prediction
    pred = ml_models["Pre-G2"].predict(df)[0]
    pred = np.round(pred,2)

    # Generate SHAP values
    shap_val = ml_models["Explainer_M2"].shap_values(df)

    return Output_schema(my_prediction=pred, 
                         shap_values=shap_val[0].tolist(),
                         feature_names=VARIABLE_ORDER_M2)


@app.post('/predict_3', response_model=Output_schema)
def predict_3(request : Input_schema_3) :
    df = pd.DataFrame([request.model_dump()])
    df = df[VARIABLE_ORDER_M3] # This step aligns the data perfectly
        
    # Apply category dtypes for XGBoost native support
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Generate Prediction
    pred = ml_models["Pre-Final"].predict(df)[0]
    pred = np.round(pred,2)

    # Generate SHAP values
    shap_val = ml_models["Explainer_M3"].shap_values(df)

    return Output_schema(my_prediction=pred, 
                         shap_values=shap_val[0].tolist(),
                         feature_names=VARIABLE_ORDER_M3)
