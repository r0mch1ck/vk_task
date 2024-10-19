import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import joblib


df_processed = pd.read_parquet('train_data_processed.parquet')

X = df_processed.drop(['id', 'label'], axis=1)
y = df_processed['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


catboost_model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    random_seed=42,
    logging_level='Silent',
    eval_metric='AUC'
)

catboost_model.fit(X, y)


model_filename = "trained_model.pkl"
joblib.dump(catboost_model, model_filename)
