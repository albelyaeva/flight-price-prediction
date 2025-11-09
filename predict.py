from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

time_map = {
    'Late_Night': 0,
    'Night': 1,
    'Early_Morning': 2,
    'Morning': 3,
    'Afternoon': 4,
    'Evening': 5
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])

    for col in ["departure_time", "arrival_time"]:
        if col in df.columns:
            df[col] = df[col].map(time_map)
            df[col] = df[col].fillna(time_map["Morning"])

    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]

    pred = model.predict(df)[0]

    return jsonify({"predicted_price": float(pred)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=True)
