from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the DictVectorizer and LogisticRegression model from the binary file
with open('model1.bin', 'rb') as file:
    model = pickle.load(file)

with open('dv.bin', 'rb') as file:
    dv = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    client_data = request.get_json()
    X = dv.transform([client_data])
    prediction = model.predict(X)
    prediction_proba = model.predict_proba(X)
    
    result = {
        'prediction': int(prediction[0]),
        'prediction_probability': prediction_proba[0].tolist()
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

# client request
# Run this request in the terminal after running the flask app
# curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d '{"job": "student", "duration": 280, "poutcome": "failure"}'

# Output
# {"prediction":0,"prediction_probability":[0.6651929652448895,0.33480703475511053]}