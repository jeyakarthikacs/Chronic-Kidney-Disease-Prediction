from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("best_model_random_forest.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # 1) Read & cast form inputs
    age    = float(request.form["age"])
    sc     = float(request.form["sc"])
    sg     = float(request.form["sg"])
    htn    = int(request.form["htn"])
    hemo   = float(request.form["hemo"])
    dm     = int(request.form["dm"])
    al     = float(request.form["al"])
    appet  = int(request.form["appet"])
    rc     = float(request.form["rc"])
    pc     = int(request.form["pc"])

    # 2) Build feature array
    X_in = np.array([[age, sc, sg, htn, hemo, dm, al, appet, rc, pc]])

    # 3) Run model
    pred  = model.predict(X_in)[0]
    proba = model.predict_proba(X_in)[0, pred]

    # 4) Prepare defaults
    message = ""
    stage   = ""
    severity= ""
    affected_percentage = ""
    image_path = ""

    # 5) Fill in according to prediction
    if pred == 0:
        # Model says "no CKD"
        message = "🎉<br>You DON'T have Chronic Kidney Disease."
        stage   = "Healthy"
        severity= "Normal"
        affected_percentage = "0%"
        image_path = "https://cdn-icons-png.flaticon.com/512/4319/4319163.png"

    else:
        # Model says "CKD" → now compute eGFR to pick Stage 1–5
        message = "Oops! 🙁<br>You have CHRONIC KIDNEY DISEASE.<br>Please consult a doctor."
        egfr = 186 * (sc ** -1.154) * (age ** -0.203)

        if egfr >= 90:
            stage = "Stage 1"
            severity = "Mild (Stage 1)"
            affected_percentage = "15%"
        elif egfr >= 60:
            stage = "Stage 2"
            severity = "Mild (Stage 2)"
            affected_percentage = "30%"
        elif egfr >= 45:
            stage = "Stage 3"
            severity = "Moderate (Stage 3)"
            affected_percentage = "60%"
        elif egfr >= 30:
            stage = "Stage 4"
            severity = "Severe (Stage 4)"
            affected_percentage = "85%"
        else:
            stage = "Stage 5"
            severity = "Kidney Failure (Stage 5)"
            affected_percentage = "100%"

        image_path = "https://cdni.iconscout.com/illustration/premium/thumb/male-doctor-and-female-doctor-4093612-3392962.png"

    # 6) Render with a **consistent** set of variables
    return render_template(
        "result.html",
        message=message,
        stage=stage,
        severity=severity,
        affected_percentage=affected_percentage,
        proba=f"{proba*100:.1f}%",
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)
