from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

fat_model = joblib.load("fat_model.pkl")
muscle_model = joblib.load("muscle_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    gender = 1 if data['gender'].lower() == 'male' else 0
    age = data['age']
    weight = data['weight']
    height = data['height']
    abdomen = data['abdomen']

    input_data = np.array([[gender, age, weight, height, abdomen]])
    fat = fat_model.predict(input_data)[0]
    muscle = muscle_model.predict(input_data)[0]

    body_type = "سمين" if fat >= 20 else "نحيف"
    plan = "تمارين تضخيم" if body_type == "نحيف" else "كارديو + مقاومة"

    return jsonify({
        "bodyFat": round(fat, 2),
        "muscle": round(muscle, 2),
        "bodyType": body_type,
        "workoutPlan": plan
    })

# لا تكتب app.run() هنا
