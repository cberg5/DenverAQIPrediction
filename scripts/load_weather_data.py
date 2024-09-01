import pandas as pd


def load_and_process_weather_data(input_path, output_path):
    # Load the weather data
    weather_df = pd.read_csv(input_path)

    # Combine 'date' and 'time' into a single 'datetime' column
    weather_df['datetime'] = pd.to_datetime(weather_df['date'] + ' ' + weather_df['time'])

    # Drop the original 'date' and 'time' columns
    weather_df.drop(columns=['date', 'time'], inplace=True)

    # Handle missing values (example: forward fill)
    weather_df.fillna(method='ffill', inplace=True)

    # Resample to daily frequency and calculate mean values for each day
    daily_weather_df = weather_df.resample('D', on='datetime').mean().reset_index()

    # Save the processed daily weather data
    daily_weather_df.to_csv(output_path, index=False)
    print(f"Processed weather data saved to: {output_path}")


if __name__ == "__main__":
    # Define file paths
    input_path = '../data/weather/denver_weather_2014_2024.csv'
    output_path = '../data/daily_denver_weather_2014_2024.csv'

    # Load and process the weather data
    load_and_process_weather_data(input_path, output_path)






