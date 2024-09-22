import pandas as pd

def merge_weather_and_aqi(weather_path, aqi_path, output_path):
    # Load the weather and AQI datasets
    weather_df = pd.read_csv(weather_path)
    aqi_df = pd.read_csv(aqi_path)

    # Ensure both date columns are datetime objects
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime']).dt.tz_localize(None)  # Remove timezone
    aqi_df['Date'] = pd.to_datetime(aqi_df['Date'])  # Ensure it's datetime without timezone

    # Merge the datasets on the date columns
    merged_df = pd.merge(weather_df, aqi_df, left_on='datetime', right_on='Date', how='outer')

    # Handle missing values
    # Example: forward fill or fill with a specific value
    merged_df.fillna(method='ffill', inplace=True)

    # Drop unnecessary columns if needed
    merged_df.drop(columns=['Date'], inplace=True)

    # Save the merged dataset
    merged_df.to_csv(output_path, index=False)
    print(f"Merged data saved to: {output_path}")

if __name__ == "__main__":
    # Define file paths
    weather_path = '/Users/cjbergin/PycharmProjects/DenverAQIPrediction/data/daily_denver_weather_2014_2024.csv'
    aqi_path = '/Users/cjbergin/PycharmProjects/DenverAQIPrediction/data/combined_aqi_2014_2024.csv'
    output_path = '/Users/cjbergin/PycharmProjects/DenverAQIPrediction/data/merged_weather_aqi_2014_2024.csv'

    # Merge the weather and AQI datasets
    merge_weather_and_aqi(weather_path, aqi_path, output_path)