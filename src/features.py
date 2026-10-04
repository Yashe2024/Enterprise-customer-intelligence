import pandas as pd
def build_features(df):
    x=df.copy()
    x["order_value_12m"]=x["orders_12m"]*x["avg_order_value"]
    x["support_rate"]=x["support_tickets_12m"]/x["tenure_months"].clip(lower=1)
    x["engagement_risk"]=x["days_since_last_order"]/(x["tenure_months"]+1)
    return x
