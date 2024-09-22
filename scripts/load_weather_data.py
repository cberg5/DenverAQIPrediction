import pandas as pd
import os

def load_and_process_weather_data(input_path, output_path):
    # Load the weather data
    weather_df = pd.read_csv(input_path)

    # Convert the 'dt_iso' column to a datetime object with specified format
    weather_df['datetime'] = pd.to_datetime(weather_df['dt_iso'], format='%Y-%m-%d %H:%M:%S %z UTC')

    # Handle missing values (example: forward fill)
    weather_df.ffill(inplace=True)

    # Select only numeric columns for resampling
    numeric_cols = weather_df.select_dtypes(include=['number']).columns
    numeric_weather_df = weather_df[['datetime'] + list(numeric_cols)]

    # Resample to daily frequency and calculate mean, max, min, and standard deviation values for each day
    daily_mean = numeric_weather_df.resample('D', on='datetime').mean().add_suffix('_mean')
    daily_max = numeric_weather_df.resample('D', on='datetime').max().add_suffix('_max')
    daily_min = numeric_weather_df.resample('D', on='datetime').min().add_suffix('_min')
    daily_std = numeric_weather_df.resample('D', on='datetime').std().add_suffix('_std')

    # Combine the features into one DataFrame
    daily_weather_df = pd.concat([daily_mean, daily_max, daily_min, daily_std], axis=1).reset_index()

    # Save the processed daily weather data
    daily_weather_df.to_csv(output_path, index=False)
    print(f"Processed weather data saved to: {output_path}")

if __name__ == "__main__":
    # Define file paths
    input_path = '/Users/cjbergin/PycharmProjects/DenverAQIPrediction/data/weather/denver_weather_2014_2024.csv'
    output_path = '/Users/cjbergin/PycharmProjects/DenverAQIPrediction/data/daily_denver_weather_2014_2024.csv'

    # Load and process the weather data
    load_and_process_weather_data(input_path, output_path)