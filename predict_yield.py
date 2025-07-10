import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv(r"E:\new demo projecct\Crop-yield-prediction-using-weather-data-and-NDVI-time-series-main\Phase3\csv\Production_with_weather_ndvi.csv")  # adjust path if needed

# Select relevant features
X = df[['Precipitation', 'avgtemp', 'NDVI']]  # using Precipitation instead of rainfall
y = df['Yield']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'random_forest_model.pkl')  # This creates the file
