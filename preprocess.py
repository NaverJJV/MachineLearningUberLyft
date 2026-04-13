import argparse
import pandas as pd


def load_and_preprocess(rides_path, weather_path):
    print(f"Loading rides data from {rides_path}...")
    rides_df = pd.read_csv(rides_path)

    print(f"Loading weather data from {weather_path}...")
    weather_df = pd.read_csv(weather_path)

    # 1. Handle missing prices in the rides dataset
    print("Dropping rows with missing prices...")
    rides_df = rides_df.dropna(subset=['price'])

    # 2. Bucket the prices into Low, Medium, High
    print("Bucketing prices into 3 categories...")
    rides_df['price_category'] = pd.qcut(rides_df['price'], q=3, labels=['Low', 'Medium', 'High'])

    # 3. Drop the original price column to prevent data leakage
    rides_df = rides_df.drop(columns=['price'])

    print("\nInitial preprocessing complete. Here is a preview of the rides data:")
    print(rides_df[['distance', 'cab_type', 'name', 'price_category']].head())

    return rides_df, weather_df


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Preprocess Uber and Lyft Cab Prices dataset.")
    parser.add_argument("--rides", type=str, required=True, help="Path to the cab_rides.csv file")
    parser.add_argument("--weather", type=str, required=True, help="Path to the weather.csv file")

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Run the preprocessing function
    clean_rides, clean_weather = load_and_preprocess(args.rides, args.weather)