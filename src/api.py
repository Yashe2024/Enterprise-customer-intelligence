from pathlib import Path
import joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI(title="Enterprise Customer Intelligence API",version="1.0.0")
MODEL=Path(__file__).resolve().parents[1]/"artifacts/gradient_boosting.joblib"

class Customer(BaseModel):
    age:int; annual_income:float; tenure_months:int; orders_12m:int
    avg_order_value:float; support_tickets_12m:int; days_since_last_order:int
    satisfaction_score:float; digital_engagement:float

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/v1/churn")
def churn(c:Customer):
    x=pd.DataFrame([c.model_dump()])
    x["order_value_12m"]=x.orders_12m*x.avg_order_value
    x["support_rate"]=x.support_tickets_12m/x.tenure_months.clip(lower=1)
    x["engagement_risk"]=x.days_since_last_order/(x.tenure_months+1)
    model=joblib.load(MODEL)
    p=float(model.predict_proba(x)[0,1])
    action="RETENTION_OFFER" if p>=.70 else "ENGAGEMENT_CAMPAIGN" if p>=.45 else "NORMAL_NURTURE"
    return {"churn_probability":round(p,4),"risk_band":"HIGH" if p>=.70 else "MEDIUM" if p>=.45 else "LOW","next_best_action":action}
