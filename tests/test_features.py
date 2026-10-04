import pandas as pd
from src.features import build_features
def test_feature_generation():
    df=pd.DataFrame({"orders_12m":[5],"avg_order_value":[20],"support_tickets_12m":[2],"tenure_months":[10],"days_since_last_order":[20]})
    x=build_features(df)
    assert x.order_value_12m.iloc[0]==100
    assert x.support_rate.iloc[0]==.2
