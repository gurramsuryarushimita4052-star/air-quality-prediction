
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Loading air quality dataset...")

df = pd.read_csv("dataset/city_day.csv")

# Keep rows where AQI is available
if "AQI" not in df.columns:
    raise ValueError("AQI column not found. Check your dataset columns.")

# Use pollutant measurements as input features
features = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene",
    "Toluene", "Xylene"
]

features = [col for col in features if col in df.columns]

if not features:
    raise ValueError("No expected pollutant columns found.")

data = df[features + ["AQI"]].copy()

for col in features + ["AQI"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna(subset=["AQI"])

X = data[features]
y = data["AQI"]

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

print("Training model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.3f}")

Path("artifacts").mkdir(exist_ok=True)

joblib.dump(model, "artifacts/aqi_model.joblib")
joblib.dump(features, "artifacts/aqi_features.joblib")

print("\nModel saved successfully!")
print("AQI MODEL TRAINING COMPLETED SUCCESSFULLY!")