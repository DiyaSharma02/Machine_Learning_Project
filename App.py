from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__, template_folder='.')

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("city_day.csv")
data = data.dropna()

# =========================
# ML MODEL
# =========================
features = ['PM2.5','PM10','NO2','CO','SO2']
target = 'AQI'

X = data[features]
y = data[target]

model = RandomForestRegressor()
model.fit(X, y)

# =========================
# ANALYSIS FUNCTION
# =========================
def generate_analysis(aqi_values):

    avg_aqi = sum(aqi_values) / len(aqi_values)
    max_aqi = max(aqi_values)
    min_aqi = min(aqi_values)

    if avg_aqi <= 50:
        trend = "Overall air quality is GOOD 😊"
    elif avg_aqi <= 100:
        trend = "Air quality is MODERATE 😐"
    else:
        trend = "Air quality is POOR 😷"

    if max_aqi > 400:
        warning = "⚠️ Severe pollution detected on some days!"
    else:
        warning = "No extreme pollution spikes."

    analysis = f"""
    <b>Average AQI:</b> {round(avg_aqi,2)} <br>
    <b>Maximum AQI:</b> {max_aqi} <br>
    <b>Minimum AQI:</b> {min_aqi} <br><br>
    <b>Status:</b> {trend} <br>
    <b>Warning:</b> {warning}
    """

    return analysis

# =========================
# HOME
# =========================
@app.route('/')
def home():
    cities = list(data['City'].unique())
    return render_template('index.html', cities=cities)

# =========================
# FILTER
# =========================
@app.route('/filter', methods=['POST'])
def filter_data():

    city = request.form['city']
    filtered = data[data['City'] == city]

    if filtered.empty:
        return "No data found"

    dates = list(filtered['Date'].astype(str))
    aqi = list(filtered['AQI'])
    pm25 = list(filtered['PM2.5'])
    pm10 = list(filtered['PM10'])

    latest_row = filtered.tail(1).to_dict(orient='records')[0]

    latest_aqi = float(latest_row['AQI'])
    pm25_val = float(latest_row['PM2.5'])
    pm10_val = float(latest_row['PM10'])
    no2_val = float(latest_row['NO2'])
    co_val = float(latest_row['CO'])
    so2_val = float(latest_row['SO2'])

    # ML Prediction
    pred = model.predict([[pm25_val, pm10_val, no2_val, co_val, so2_val]])[0]
    pred = float(pred)

    # =========================
    # HEALTH ADVICE + PRECAUTIONS
    # =========================
    if pred <= 50:
        advice = "Good 😊 Safe to go outside"
        precautions = [
            "Enjoy outdoor activities",
            "No health risk"
        ]
        color = "green"

    elif pred <= 100:
        advice = "Moderate 😐 Limit outdoor activities"
        precautions = [
            "Reduce prolonged outdoor exposure",
            "Sensitive people should be careful"
        ]
        color = "orange"

    else:
        advice = "Poor 😷 Health Risk!"
        precautions = [
            "Stay indoors",
            "Wear N95 mask if going outside",
            "Avoid exercise",
            "Use air purifier"
        ]
        color = "red"

    # ANALYSIS
    analysis = generate_analysis(aqi)

    return render_template(
        'index.html',
        cities=list(data['City'].unique()),
        dates=dates,
        aqi=aqi,
        pm25=pm25,
        pm10=pm10,
        latest_aqi=latest_aqi,
        pm25_val=pm25_val,
        pm10_val=pm10_val,
        pred=round(pred, 2),
        advice=advice,
        color=color,
        analysis=analysis,
        precautions=precautions
    )

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
