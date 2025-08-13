import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

def load_data(url, sample_size=50000):
    print("\n[INFO] Loading dataset...")
    try:
        df = pd.read_csv(url).sample(n=sample_size, random_state=42)
        print(f"[SUCCESS] Loaded {len(df)} rows from dataset.")
        return df
    except Exception as e:
        print(f"[ERROR] Could not load dataset: {e}")
        return None

def clean_data(df):
    print("\n[INFO] Cleaning and preparing dataset...")
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df.dropna(subset=['Start_Time'], inplace=True)
    df['Hour'] = df['Start_Time'].dt.hour
    df['DayOfWeek'] = df['Start_Time'].dt.day_name()
    df['Month'] = df['Start_Time'].dt.month_name()
    df.dropna(subset=['Start_Lat', 'Start_Lng'], inplace=True)
    df = df[(df['Start_Lat'].between(-90, 90)) & (df['Start_Lng'].between(-180, 180))]
    print(f"[INFO] Cleaned dataset now has {len(df)} rows.")
    return df

def explore_data(df):
    print("\n[INFO] Dataset Overview:")
    print(df.head(3))
    print("\n[INFO] Dataset Shape:", df.shape)
    print("\n[INFO] Missing Values:\n", df.isnull().sum().head(10))
    print("\n[INFO] Data Types:\n", df.dtypes.head(10))

def plot_accidents_by_hour(df):
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Hour', data=df, palette='viridis')
    plt.title('Accidents by Hour of the Day', fontsize=16)
    plt.xlabel('Hour of the Day (0-23)')
    plt.ylabel('Number of Accidents')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('accidents_by_hour.png')
    print("[SUCCESS] Saved plot: accidents_by_hour.png")
    plt.close()

def plot_accidents_by_weekday(df):
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plt.figure(figsize=(12, 6))
    sns.countplot(x='DayOfWeek', data=df, order=order, palette='cubehelix')
    plt.title('Accidents by Day of the Week', fontsize=16)
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Accidents')
    plt.savefig('accidents_by_weekday.png')
    print("[SUCCESS] Saved plot: accidents_by_weekday.png")
    plt.close()

def plot_accidents_by_month(df):
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Month', data=df, order=month_order, palette='mako')
    plt.title('Accidents by Month', fontsize=16)
    plt.xticks(rotation=45)
    plt.savefig('accidents_by_month.png')
    print("[SUCCESS] Saved plot: accidents_by_month.png")
    plt.close()

def plot_weather_conditions(df):
    top_weather = df['Weather_Condition'].value_counts().nlargest(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(y=top_weather.index, x=top_weather.values, palette='plasma', orient='h')
    plt.title('Top 10 Weather Conditions During Accidents', fontsize=16)
    plt.xlabel('Number of Accidents')
    plt.ylabel('Weather Condition')
    plt.savefig('accidents_by_weather.png')
    print("[SUCCESS] Saved plot: accidents_by_weather.png")
    plt.close()

def plot_severity_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Severity', data=df, palette='coolwarm')
    plt.title('Accident Severity Distribution', fontsize=16)
    plt.xlabel('Severity Level')
    plt.ylabel('Number of Accidents')
    plt.savefig('accidents_severity.png')
    print("[SUCCESS] Saved plot: accidents_severity.png")
    plt.close()

def create_hotspot_map(df):
    print("\n[INFO] Generating accident hotspot map...")
    map_center = [39.8283, -98.5795]
    accident_map = folium.Map(location=map_center, zoom_start=4)
    heat_data = df[['Start_Lat', 'Start_Lng']].values.tolist()
    HeatMap(heat_data, radius=8, blur=6).add_to(accident_map)
    accident_map.save("accident_hotspot_map.html")
    print("[SUCCESS] Saved map: accident_hotspot_map.html")

def analyze_traffic_accidents():
    print("\n===== TRAFFIC ACCIDENT ANALYSIS STARTED =====")
    url = 'https://raw.githubusercontent.com/adyngom/US-Accidents-EDA/master/US_Accidents_Dec21_updated.csv'
    df = load_data(url)
    if df is None:
        return
    explore_data(df)
    df = clean_data(df)
    plot_accidents_by_hour(df)
    plot_accidents_by_weekday(df)
    plot_accidents_by_month(df)
    plot_weather_conditions(df)
    plot_severity_distribution(df)
    create_hotspot_map(df)
    print("\n===== TRAFFIC ACCIDENT ANALYSIS COMPLETED =====")

if __name__ == '__main__':
    analyze_traffic_accidents()
