import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import joblib

df_processed = pd.read_parquet('test_data_processed.parquet')

X = df_processed.drop(['id'], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

catboost_model = joblib.load("trained_model.pkl")

y_pred_prob_catboost = catboost_model.predict_proba(X)[:, 1]

result_df = pd.DataFrame({'id': df_processed['id'], 'score': y_pred_prob_catboost})

result_df.to_csv('submissions.csv', index=False)
