import pickle
from flask import Flask
from flask import request
from flask import jsonify
from train import preprocess, make_features, features
import pandas as pd

model_file = 'xgb_classifier.bin'

def load_model(path):
        
    with open(model_file, 'rb') as f_in:
        model = pickle.load(f_in)
    return model

app = Flask('sleep-state')

@app.route('/predict', methods=['POST'])
def predict():
    child = request.get_json()
    print('child')
    print(child)
    # child_df=pd.DataFrame([child])
    child_df = pd.DataFrame(child)
    print('child_df')
    
    # print(child_df)
    child_df=preprocess(child_df)
    child_df=make_features(child_df)
    X = child_df[features]
    print(X)

    model=load_model(model_file)
    # for
    y_pred = model.predict_proba(X)[0, 1]
    print('y_pred',type(y_pred))
    print('y_pred',y_pred)
    

    awake = y_pred >= 0.5

    result = {
        'awake_probabilities': float(y_pred),
        'awake': bool(awake)  
    }
    print("result",result)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)