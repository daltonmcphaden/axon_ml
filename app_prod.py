import json
from flask import Flask, request, jsonify
from Predict_prod import generate_predictions
from Train_prod import train_model
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    cwd = os.getcwd()
    return cwd

@app.route('/train', methods=['POST'])
def train():  
    req = request.get_json()  
    model_id = train_model(req["modelName"], req["epochs"], req["learning_rate"], req["slidingWindow"], req["modelType"], req["caseIDs"])
    return jsonify({"model_id": model_id})

@app.route('/predict', methods=['POST'])
def predict():
    print("Predicting...")
    req = request.get_json()
    batchName = generate_predictions(req["batchName"], req["caseIDs"], req["modelID"])    
    return jsonify({"batchName": batchName})

if __name__ == "__main__":
    app.run(debug=True)