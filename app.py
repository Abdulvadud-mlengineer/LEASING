# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# CORS — чтобы Lovable мог обращаться к API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружаем модель
model = joblib.load('credit_scoring_model.pkl')
features = joblib.load('model_features.pkl')

class ClientData(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    CODE_GENDER: int      # 0=M, 1=F
    FLAG_OWN_CAR: int     # 0=N, 1=Y
    FLAG_OWN_REALTY: int  # 0=N, 1=Y

@app.post("/predict")
def predict(data: ClientData):
    # Создаём пустой датафрейм со всеми фичами
    input_df = pd.DataFrame(0, index=[0], columns=features)
    
    # Заполняем поля которые пришли
    for field, value in data.dict().items():
        if field in input_df.columns:
            input_df[field] = value
    
    # Предсказание
    proba = model.predict_proba(input_df)[0][1]
    score = int((1 - proba) * 1000)  # скоринговый балл 0-1000
    
    risk = "🟢 Низкий риск" if proba < 0.3 else "🟡 Средний риск" if proba < 0.6 else "🔴 Высокий риск"
    
    return {
        "probability": round(proba, 4),
        "score": score,
        "risk": risk
    }

@app.get("/")
def root():
    return {"status": "Credit Scoring API работает!"}