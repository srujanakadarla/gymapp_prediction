
import flask
from flask import Flask, request , jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import requests
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
filename = 'Model_GYM.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route('/',methods=['GET'])
def main():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    arr = np.array([[data1, data2, data3]])
    pred = model.predict(arr)
    return render_template('after.html',data=pred)


if __name__ == "__main__":
    app.run(debug=True)
