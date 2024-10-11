import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from google.cloud import storage
import io

def load_data_from_gcs(bucket_name, file_path):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    data = blob.download_as_string()
    return pd.read_csv(io.StringIO(data.decode('utf-8')))

def save_model_to_gcs(model, bucket_name, file_path):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    model_data = io.BytesIO()
    joblib.dump(model, model_data)
    model_data.seek(0)
    blob.upload_from_file(model_data, content_type='application/octet-stream')

def prepare_data(df):
    df['datetime'] = pd.to_datetime(df['datetime'])

    df['season'] = df['datetime'].dt.month % 12 // 3 + 1
    df['day_of_year'] = df['datetime'].dt.dayofyear

    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    df['AQI_lag_1'] = df['AQI Value'].shift(1)
    df['AQI_lag_3'] = df['AQI Value'].shift(3)

    df['temp_mean_7d_avg'] = df['temp_mean'].rolling(window=7).mean()
    df['humidity_mean_7d_avg'] = df['humidity_mean'].rolling(window=7).mean()

    df['temp_mean_squared'] = df['temp_mean'] ** 2

    df = df.apply(pd.to_numeric, errors='coerce')

    df = df.dropna(subset=['AQI Value', 'temp_mean_7d_avg', 'AQI_lag_1'])

    features = [
        'temp_mean', 'temp_max_mean', 'temp_min_mean', 'wind_speed_mean', 'humidity_mean',
        'pressure_mean', 'clouds_all_mean', 'season', 'day_of_year', 'day_of_week', 'is_weekend',
        'AQI_lag_1', 'AQI_lag_3', 'temp_mean_7d_avg', 'humidity_mean_7d_avg', 'temp_mean_squared'
    ]

    X = df[features]
    y = df['AQI Value']

    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_random_forest_with_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 500, 1000],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    rf = RandomForestRegressor(random_state=42)
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
                                   n_iter=20, cv=5, verbose=2, random_state=42, n_jobs=-1)

    rf_random.fit(X_train, y_train)

    print("Best parameters found:", rf_random.best_params_)
    return rf_random.best_estimator_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R² Score: {r2}")


if __name__ == "__main__":
    bucket_name = 'weather-aqi-data-storage'
    data_path = 'merged_weather_aqi_2014_2024.csv'
    model_save_path = 'models/trained_model.pkl'

    df = load_data_from_gcs(bucket_name, data_path)

    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training Random Forest model with hyperparameter tuning...")
    best_rf_model = train_random_forest_with_tuning(X_train, y_train)
    print("Best Random Forest Model training completed.")

    evaluate_model(best_rf_model, X_test, y_test)

    save_model_to_gcs(best_rf_model, bucket_name, model_save_path)
    print(f"Model saved successfully to 'gs://{bucket_name}/{model_save_path}'.")