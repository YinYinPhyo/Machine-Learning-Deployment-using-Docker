# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:50:04 2020
@author: pramod.singh """
from flask import Flask, request 
import numpy as np
import pickle
import pandas as pd
from flasgger import Swagger

app = Flask(__name__) 
Swagger(app)

pickle_in = open("logreg.pkl", "rb") 
model = pickle.load(pickle_in)

@app.route('/') 
def home():
    return "Welcome to the Flask API!"

@app.route('/predict', methods=["GET"])
def predict_class():
    """Predict if Customer would buy the product or not.
    ---
    parameters:
      - name: age
        in: query
        type: integer
        required: true
        description: Age of the customer
      - name: new_user
        in: query
        type: integer
        required: true
        description: Is the customer a new user (0 or 1)?
      - name: total_pages_visited
        in: query
        type: integer
        required: true
        description: Number of pages visited
    responses:
      200:
        description: Prediction result
        schema:
          type: string
    """
    age = int(request.args.get("age"))
    new_user = int(request.args.get("new_user"))
    total_pages_visited = int(request.args.get("total_pages_visited"))
    prediction = model.predict([[age, new_user, total_pages_visited]])
    return "Model prediction is " + str(prediction)

@app.route('/predict_file', methods=["POST"])
def prediction_test_file():
    """Prediction on multiple input test file.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Upload a CSV file with test data
    responses:
      200:
        description: Predictions for each row in the test file
        schema:
          type: array
          items:
            type: number
    """
    df_test = pd.read_csv(request.files.get("file"))
    prediction = model.predict(df_test)
    return str(list(prediction))


@app.route('/apidocs')
def api_docs():
	return "API Documentation goes here."
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    
