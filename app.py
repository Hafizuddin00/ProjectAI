from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("C:/Users/muhdh/Desktop/AQI_Predictor/globalAirPollutionDatasetOriginal.csv")

app = Flask(__name__)

def train_model(data):
    relevant_features = [
        'CO AQI Value', 'CO AQI Category', 'Ozone AQI Value', 'Ozone AQI Category',
        'NO2 AQI Value', 'NO2 AQI Category', 'PM2.5 AQI Value', 'PM2.5 AQI Category',
        'AQI Category'
    ]
    relevant_data = data[relevant_features]
    predictor = relevant_data.columns[:-1]
    target = relevant_data.columns[-1]

    label_encoder = LabelEncoder()
    for column in relevant_data.select_dtypes(include=['object']).columns:
        relevant_data[column] = label_encoder.fit_transform(relevant_data[column])

    train, test = train_test_split(relevant_data, test_size=0.3, random_state=0, stratify=relevant_data[target])
    model = RandomForestClassifier(random_state=0)
    model.fit(train[predictor], train[target])

    return model, label_encoder

model, label_encoder = train_model(data)

def predict_aqi_category(model, label_encoder, co_aqi, co_aqi_category, ozone_aqi, ozone_aqi_category, no2_aqi, no2_aqi_category, pm25_aqi, pm25_aqi_category):
    try:
        co_aqi_category_encoded = label_encoder.transform([co_aqi_category])[0]
        ozone_aqi_category_encoded = label_encoder.transform([ozone_aqi_category])[0]
        no2_aqi_category_encoded = label_encoder.transform([no2_aqi_category])[0]
        pm25_aqi_category_encoded = label_encoder.transform([pm25_aqi_category])[0]

        user_input = pd.DataFrame({
            'CO AQI Value': [co_aqi],
            'CO AQI Category': [co_aqi_category_encoded],
            'Ozone AQI Value': [ozone_aqi],
            'Ozone AQI Category': [ozone_aqi_category_encoded],
            'NO2 AQI Value': [no2_aqi],
            'NO2 AQI Category': [no2_aqi_category_encoded],
            'PM2.5 AQI Value': [pm25_aqi],
            'PM2.5 AQI Category': [pm25_aqi_category_encoded]
        })

        prediction = model.predict(user_input)
        predicted_category = label_encoder.inverse_transform(prediction)[0]
        return predicted_category
    except ValueError:
        return 'Invalid input for AQI categories'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    co_aqi = float(data['co_aqi'])
    ozone_aqi = float(data['ozone_aqi'])
    no2_aqi = float(data['no2_aqi'])
    pm25_aqi = float(data['pm25_aqi'])

    # Determine AQI categories based on ranges
    co_category = determine_category(co_aqi, [0, 50, 100, 150, 200, 300, 500])
    ozone_category = determine_category(ozone_aqi, [0, 50, 100, 150, 200, 300, 500])
    no2_category = determine_category(no2_aqi, [0, 50, 100, 150, 200, 300, 500])
    pm25_category = determine_category(pm25_aqi, [0, 50, 100, 150, 200, 300, 500])

    predicted_category = predict_aqi_category(
        model, label_encoder, co_aqi, co_category, ozone_aqi, ozone_category,
        no2_aqi, no2_category, pm25_aqi, pm25_category
    )
    return jsonify({
        'predicted_category': predicted_category,
        'co_category': co_category,
        'ozone_category': ozone_category,
        'no2_category': no2_category,
        'pm25_category': pm25_category
    })

def determine_category(value, ranges):
    for i in range(len(ranges) - 1):
        if ranges[i] <= value <= ranges[i + 1]:
            category_names = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
            return category_names[i]
    return 'Undefined Category'

if __name__ == '__main__':
    app.run(debug=True)
