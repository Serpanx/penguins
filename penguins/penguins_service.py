import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

def predict_single(penguin, sc, dv, model):
    penguin = pd.DataFrame(penguin,index=[True])
    categorical = ["island", "sex"]
    numerical = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
    X_dict = penguin[categorical].to_dict(orient='records')
    X_one_hot = dv.transform(X_dict)
    X_std = sc.transform(penguin[numerical])
    X = np.hstack((X_one_hot, X_std))
    y_pred = model.predict(X)[0]
    y_prob = model.predict_proba(X)[0]
    return (y_pred,y_prob)

def predict(sc, dv, model):
    penguin = request.get_json()
    race, probability = predict_single(penguin, sc, dv, model)
    res=''
    if np.array_equal(race, 0):
        res = ('Adelie')
    elif np.array_equal(race, 1):
        res = ('Chinstrap')
    elif np.array_equal(race, 2):
        res = ('Gentoo')
    else: res = ('unknown')

    probability = round((float(max(probability)*100)),2)

    result = {
        'penguin': str(res),
        'accuracy(%)': str(probability)
    }

    return jsonify(result)

app = Flask('penguin')

@app.route('/predict_lr', methods=['POST'])
def predict_lr():
    with open('models/lr.pck', 'rb') as f:
        sc, dv, model = pickle.load(f)
    return predict(sc, dv, model)

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    with open('models/svm.pck', 'rb') as f:
        sc, dv, model = pickle.load(f)
    return predict(sc, dv, model)

@app.route('/predict_dt', methods=['POST'])
def predict_dt():
    with open('models/dt.pck', 'rb') as f:
        sc, dv, model = pickle.load(f)
    return predict(sc, dv, model)

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    with open('models/knn.pck', 'rb') as f:
        sc, dv, model = pickle.load(f)
    return predict(sc, dv, model)

if __name__ == '__main__':
    app.run(debug=True, port=8000)