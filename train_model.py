import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
import joblib

def generate_weather_data(n_samples=5000, random_seed=42):
    np.random.seed(random_seed)
    temp = np.random.normal(loc=27.0, scale=4.5, size=n_samples)
    humidity = np.random.uniform(low=40.0, high=98.0, size=n_samples)
    pressure = np.random.normal(loc=1010.0, scale=7.0, size=n_samples)
    wind_speed = np.random.exponential(scale=12.0, size=n_samples) + 2.0
    cloud_cover = np.random.uniform(low=10.0, high=100.0, size=n_samples)
    dew_point = temp - ((100 - humidity) / 5)

    precip_score = (
        0.55 * (humidity - 50)
        + 0.35 * cloud_cover
        + 0.25 * wind_speed
        - 0.80 * (pressure - 1005)
        + np.random.normal(0, 10, size=n_samples)
    )
    rainfall_mm = np.maximum(0.0, np.where(precip_score > 10, (precip_score - 10) * 1.6, 0.0))

    def assign_risk(rf):
        if rf < 2.5:
            return "No Rain / Trace"
        elif rf < 35.5:
            return "Moderate Risk"
        elif rf < 64.5:
            return "Heavy Risk"
        else:
            return "Extreme / Very Heavy Risk"

    risk_category = [assign_risk(r) for r in rainfall_mm]

    return pd.DataFrame({
        'temperature': np.clip(temp, 10, 48),
        'humidity': np.clip(humidity, 10, 100),
        'pressure': np.clip(pressure, 970, 1040),
        'wind_speed': np.clip(wind_speed, 0, 90),
        'cloud_cover': np.clip(cloud_cover, 0, 100),
        'dew_point': np.clip(dew_point, 5, 38),
        'rainfall_mm': rainfall_mm,
        'risk_category': risk_category
    })

def train_and_export():
    print("Generating meteorological dataset...")
    df = generate_weather_data()
    feature_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'cloud_cover', 'dew_point']
    
    X = df[feature_cols]
    y_reg = df['rainfall_mm']
    y_cls = df['risk_category']

    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Regressor...")
    regressor = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    regressor.fit(X_train_scaled, y_reg_train)

    print("Training Classifier...")
    classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    classifier.fit(X_train_scaled, y_cls_train)

    bundle = {
        'scaler': scaler,
        'regressor': regressor,
        'classifier': classifier,
        'features': feature_cols,
        'classes': classifier.classes_.tolist()
    }
    joblib.dump(bundle, 'rainfall_pipeline.pkl')
    print("Success: rainfall_pipeline.pkl created!")

if __name__ == '__main__':
    train_and_export()