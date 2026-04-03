# Import required libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS (frontend can access API)

# Load saved model and encoder
model = joblib.load('sales_model.pkl')
le = joblib.load('label_encoder.pkl')

# Example festival dates (you can modify)
festival_list = [
    "2025-01-14",  # Pongal
    "2025-11-01",  # Diwali (example)
    "2025-12-25",  # Christmas
    "2025-04-10",  # Ramzan (example)
]

# Home route (optional)
@app.route('/')
def home():
    return "Food Demand Prediction API is running!"

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        item = data['item']
        date_str = data['date']

        # Convert string to datetime
        date = datetime.strptime(date_str, "%Y-%m-%d")

        # Extract features
        day = date.day
        month = date.month
        year = date.year
        weekday = date.weekday()  # Monday=0, Sunday=6

        # Weekend check
        is_weekend = 1 if weekday >= 5 else 0

        # Festival check
        is_festival = 1 if date_str in festival_list else 0

        # Encode item
        item_encoded = le.transform([item])[0]

        # Create DataFrame (same format as training)
        input_data = pd.DataFrame([{
            'item_encoded': item_encoded,
            'day': day,
            'month': month,
            'year': year,
            'weekday': weekday,
            'is_weekend': is_weekend,
            'is_festival': is_festival
        }])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Return response
        return jsonify({
            "item": item,
            "predicted_sales": int(prediction)
        })

    except Exception as e:
        # Error handling
        return jsonify({
            "error": str(e)
        })

# Run app
if __name__ == '__main__':
    app.run(debug=True, port=5000)   #/home/karthiga/Documents/pythonbykarthiga/sales-project/food-demand-api/sales_model.pkl