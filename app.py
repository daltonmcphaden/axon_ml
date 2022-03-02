from flask import Flask, request, jsonify
from Predict2 import generate_predictions
from Train2 import train_model

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return "<h1>Test Flask App</h1>"

@app.route('/train', methods=['GET'])
def train():
    # print(request.method)
    # print(request.url)
    # print(request.args)
    train_model()
    return "<h1>Test Flask App</h1>"

@app.route('/predict', methods=['GET'])
def predict():
    predictions = generate_predictions()
    return jsonify({"predictions": predictions.tolist()})

# The return type must be a string, dict, tuple, Response instance, or WSGI callable, but it was a ndarray