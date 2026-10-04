import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from .features import build_features

BASE=Path(__file__).resolve().parents[1]
df=build_features(pd.read_csv(BASE/"data/customer_360.csv"))
target="churned"
features=["age","annual_income","tenure_months","orders_12m","avg_order_value",
          "support_tickets_12m","days_since_last_order","satisfaction_score",
          "digital_engagement","order_value_12m","support_rate","engagement_risk"]
X=df[features]; y=df[target]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
models={
 "logistic":LogisticRegression(max_iter=1000),
 "random_forest":RandomForestClassifier(n_estimators=250,max_depth=10,random_state=42,n_jobs=-1),
 "gradient_boosting":GradientBoostingClassifier(random_state=42)
}
results={}
for name,model in models.items():
    pipe=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",model)])
    pipe.fit(Xtr,ytr)
    prob=pipe.predict_proba(Xte)[:,1]
    pred=(prob>=.5).astype(int)
    results[name]={
        "roc_auc":round(float(roc_auc_score(yte,prob)),4),
        "precision":round(float(precision_score(yte,pred)),4),
        "recall":round(float(recall_score(yte,pred)),4),
        "f1":round(float(f1_score(yte,pred)),4)
    }
    import joblib
    Path(BASE/"artifacts").mkdir(exist_ok=True)
    joblib.dump(pipe,BASE/f"artifacts/{name}.joblib")
best=max(results,key=lambda k:results[k]["roc_auc"])
Path(BASE/"artifacts/metrics.json").write_text(json.dumps({"models":results,"champion":best},indent=2))
print(json.dumps({"champion":best,"metrics":results},indent=2))
