
import pandas as pd

df = pd.read_csv("dataset/city_day.csv")

print("AIR QUALITY DATASET LOADED SUCCESSFULLY!")

print("\nFIRST FIVE ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nAQI SUMMARY")
if "AQI" in df.columns:
    print(df["AQI"].describe())

print("\nCITY NAMES")
if "City" in df.columns:
    print(df["City"].unique())

print("\nEDA COMPLETED SUCCESSFULLY!")