import os
from flask import Flask, request, jsonify
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# تحميل النماذج من ملفات محليّة ضمن الريبو
fat_model = joblib.load(os.path.join(BASE_DIR, "fat_model.pkl"))
muscle_model = joblib.load(os.path.join(BASE_DIR, "muscle_model.pkl"))

app = Flask(__name__)

@app.get("/")
def health():
    return {"status": "ok"}, 200

@app.post("/predict")
def predict():
    try:
        data = request.get_json(force=True)

        gender = 1 if str(data["gender"]).lower() == "male" else 0
        age = float(data["age"])
        weight = float(data["weight"])
        height = float(data["height"])
        abdomen = float(data["abdomen"])

        input_data = np.array([[gender, age, weight, height, abdomen]])
        fat = float(fat_model.predict(input_data)[0])
        muscle = float(muscle_model.predict(input_data)[0])

        body_type = "سمين" if fat >= 20 else "نحيف"
        plan = "تمارين تضخيم (أوزان + بروتين)" if body_type == "نحيف" else "كارديو + مقاومة لتقليل الدهون"

        return jsonify({
            "bodyFat": round(fat, 2),
            "muscle": round(muscle, 2),
            "bodyType": body_type,
            "workoutPlan": plan
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Render يمرّر البورت في متغير PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
