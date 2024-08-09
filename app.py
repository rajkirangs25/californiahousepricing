import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

## Load the Model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    try:
        data = request.json['data']
        print(f"Received data: {data}")
        data_array = np.array(list(data.values())).reshape(1, -1)
        print(f"Data array: {data_array}")
        new_data = scaler.transform(data_array)
        print(f"Scaled data: {new_data}")
        prediction = regmodel.predict(new_data)
        print(f"Prediction: {prediction[0]}")
        # Convert the prediction to a Python data type
        prediction_output = prediction[0].tolist()
        return jsonify({'prediction': prediction_output})
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 400
    
@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template('home.html', prediction_text = f"The House price prediction is {output}")

if __name__ == '__main__':
    app.run(debug = True)